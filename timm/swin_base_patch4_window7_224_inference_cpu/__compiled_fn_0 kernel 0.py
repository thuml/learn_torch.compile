
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (48L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (48L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp15);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr4[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(128L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x4 + (128L*x3) + (896L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr3[static_cast<long>(x3 + (7L*x2) + (56L*x1) + (392L*x0))];
                                auto tmp4 = out_ptr4[static_cast<long>(x3 + (7L*x2) + (56L*x1) + (392L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(128.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr5 + static_cast<long>(x4 + (128L*x3) + (896L*x1) + (6272L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (384L*x2) + (18816L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (6272L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (18816L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (6272L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (18816L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (6272L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (9604L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (4L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (196L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (9604L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (4L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (9604L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(256L + x3 + (32L*x1) + (384L*x2) + (18816L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (6272L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (6272L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (6272L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>(c10::div_floor_integer(x1, 56L)) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 392L))) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (3136L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (3136L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>(c10::div_floor_integer(x1, 56L)) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 392L))) + (401408L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (3136L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (3136L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(128.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(7L))) + (896L*(static_cast<long>(x1) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (401408L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (56L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(128L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (128L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((56L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(56L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L))) + (401408L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (128L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((56L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(56L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L))) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((56L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(56L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L)), 7L))) + (401408L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (128L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(56L))) + (401408L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((56L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(56L))) + (3136L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((56L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(56L))) + (3136L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(56L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(128.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (128L*x4) + (896L*x2) + (6272L*x3) + (50176L*x1) + (401408L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (384L*x2) + (18816L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (6272L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (18816L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (6272L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (18816L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (6272L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (9604L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(64L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (4L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (196L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (9604L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(64L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (4L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (9604L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(256L + x3 + (32L*x1) + (384L*x2) + (18816L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (6272L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (6272L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (128L*x1) + (6272L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*(static_cast<long>(x1) % static_cast<long>(56L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((56L*(c10::div_floor_integer(x1, 56L))) + (static_cast<long>(x1) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L))) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((56L*(c10::div_floor_integer(x1, 56L))) + (static_cast<long>(x1) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L))) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((56L*(c10::div_floor_integer(x1, 56L))) + (static_cast<long>(x1) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L)), 7L))) + (401408L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*(static_cast<long>((static_cast<long>((53L + (static_cast<long>(x1) % static_cast<long>(56L)))) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>((static_cast<long>((53L + (c10::div_floor_integer(x1, 56L)))) % static_cast<long>(56L))) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>((53L + (static_cast<long>(x1) % static_cast<long>(56L)))) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((53L + (c10::div_floor_integer(x1, 56L)))) % static_cast<long>(56L)), 7L))) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (3136L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (3136L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*(static_cast<long>(x1) % static_cast<long>(56L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((56L*(c10::div_floor_integer(x1, 56L))) + (static_cast<long>(x1) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L))) + (401408L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((56L*(c10::div_floor_integer(x1, 56L))) + (static_cast<long>(x1) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L))) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((56L*(c10::div_floor_integer(x1, 56L))) + (static_cast<long>(x1) % static_cast<long>(56L))), 56L)) % static_cast<long>(56L)), 7L))) + (401408L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (128L*(static_cast<long>((static_cast<long>((53L + (static_cast<long>(x1) % static_cast<long>(56L)))) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>((static_cast<long>((53L + (c10::div_floor_integer(x1, 56L)))) % static_cast<long>(56L))) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>((53L + (static_cast<long>(x1) % static_cast<long>(56L)))) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((53L + (c10::div_floor_integer(x1, 56L)))) % static_cast<long>(56L)), 7L))) + (401408L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (3136L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (3136L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(128.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(7L))) + (896L*(static_cast<long>(x1) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (401408L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (128L*(static_cast<long>((static_cast<long>((53L + x2)) % static_cast<long>(56L))) % static_cast<long>(7L))) + (896L*(static_cast<long>((static_cast<long>((53L + x1)) % static_cast<long>(56L))) % static_cast<long>(7L))) + (6272L*(c10::div_floor_integer((static_cast<long>((53L + x2)) % static_cast<long>(56L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((53L + x1)) % static_cast<long>(56L)), 7L))) + (401408L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            tmp8.store(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((128L*(c10::div_floor_integer(x2, 256L))) + (256L*x1) + (256L*x1_inner) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 128L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (28L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (28L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>((128L*(c10::div_floor_integer(x2, 256L))) + (256L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 128L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(128L)))];
                            tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                        }
                        out_ptr0[static_cast<long>(x1 + (28L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (28L*x0))] = tmp_acc0.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>((128L*(c10::div_floor_integer((x2 + x2_inner), 256L))) + (256L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 128L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = out_ptr0[static_cast<long>(x1 + (28L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x1 + (28L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(512.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (14336L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(256L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (256L*x3) + (1792L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (28L*x1) + (196L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (28L*x1) + (196L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(256.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (256L*x3) + (1792L*x1) + (12544L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (768L*x2) + (37632L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(256L + x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x1 + (768L*x2) + (37632L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (12544L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (8L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (8L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(512L + x3 + (32L*x1) + (768L*x2) + (37632L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (12544L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>(c10::div_floor_integer(x1, 28L)) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 196L))) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>(c10::div_floor_integer(x1, 28L)) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 196L))) + (200704L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (784L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(256.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(7L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (200704L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(4L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(256L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (256L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((28L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(28L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L))) + (200704L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (256L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((28L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(28L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L))) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((28L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(28L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L)), 7L))) + (200704L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (256L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(28L))) + (200704L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((28L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(28L))) + (784L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((28L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(28L))) + (784L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(28L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(256.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (256L*x4) + (1792L*x2) + (12544L*x3) + (50176L*x1) + (200704L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (768L*x2) + (37632L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(256L + x1 + (768L*x2) + (37632L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x1 + (768L*x2) + (37632L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (12544L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(16L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (8L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(16L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (8L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(512L + x3 + (32L*x1) + (768L*x2) + (37632L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (12544L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (256L*x1) + (12544L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*(static_cast<long>(x1) % static_cast<long>(28L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((28L*(c10::div_floor_integer(x1, 28L))) + (static_cast<long>(x1) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L))) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((28L*(c10::div_floor_integer(x1, 28L))) + (static_cast<long>(x1) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L))) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((28L*(c10::div_floor_integer(x1, 28L))) + (static_cast<long>(x1) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L)), 7L))) + (200704L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*(static_cast<long>((static_cast<long>((25L + (static_cast<long>(x1) % static_cast<long>(28L)))) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>((static_cast<long>((25L + (c10::div_floor_integer(x1, 28L)))) % static_cast<long>(28L))) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>((25L + (static_cast<long>(x1) % static_cast<long>(28L)))) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((25L + (c10::div_floor_integer(x1, 28L)))) % static_cast<long>(28L)), 7L))) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*(static_cast<long>(x1) % static_cast<long>(28L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((28L*(c10::div_floor_integer(x1, 28L))) + (static_cast<long>(x1) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L))) + (200704L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((28L*(c10::div_floor_integer(x1, 28L))) + (static_cast<long>(x1) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L))) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((28L*(c10::div_floor_integer(x1, 28L))) + (static_cast<long>(x1) % static_cast<long>(28L))), 28L)) % static_cast<long>(28L)), 7L))) + (200704L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*(static_cast<long>((static_cast<long>((25L + (static_cast<long>(x1) % static_cast<long>(28L)))) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>((static_cast<long>((25L + (c10::div_floor_integer(x1, 28L)))) % static_cast<long>(28L))) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>((25L + (static_cast<long>(x1) % static_cast<long>(28L)))) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((25L + (c10::div_floor_integer(x1, 28L)))) % static_cast<long>(28L)), 7L))) + (200704L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (784L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(256.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(7L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (200704L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*(static_cast<long>((static_cast<long>((25L + x2)) % static_cast<long>(28L))) % static_cast<long>(7L))) + (1792L*(static_cast<long>((static_cast<long>((25L + x1)) % static_cast<long>(28L))) % static_cast<long>(7L))) + (12544L*(c10::div_floor_integer((static_cast<long>((25L + x2)) % static_cast<long>(28L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((25L + x1)) % static_cast<long>(28L)), 7L))) + (200704L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            tmp8.store(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>((256L*(c10::div_floor_integer(x2, 512L))) + (512L*x1) + (512L*x1_inner) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 256L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (14L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (14L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>((256L*(c10::div_floor_integer(x2, 512L))) + (512L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 256L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(256L)))];
                            tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                        }
                        out_ptr0[static_cast<long>(x1 + (14L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (14L*x0))] = tmp_acc0.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>((256L*(c10::div_floor_integer((x2 + x2_inner), 512L))) + (512L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 256L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = out_ptr0[static_cast<long>(x1 + (14L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x1 + (14L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(1024.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (1024L*x1) + (14336L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_native_layer_norm_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(7L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(512L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (512L*x3) + (3584L*x2) + (7168L*x1) + (50176L*x0)));
                                auto tmp1 = out_ptr0[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp4 = out_ptr1[static_cast<long>(x3 + (7L*x2) + (14L*x1) + (98L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x4));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x4));
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 - tmp2;
                                auto tmp5 = static_cast<float>(512.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = 1 / std::sqrt(tmp8);
                                auto tmp10 = at::vec::Vectorized<float>(tmp9);
                                auto tmp11 = tmp3 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x4 + (512L*x3) + (3584L*x1) + (25088L*x2) + (50176L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>(c10::div_floor_integer(x1, 14L)) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer(x1, 98L))) + (100352L*x0)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(512.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(7L); x4+=static_cast<long>(1L))
                            {
                                for(long x5=static_cast<long>(0L); x5<static_cast<long>(512L); x5+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x5 + (512L*(static_cast<long>((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x5 + (512L*(static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L))) + (7168L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (100352L*x0)));
                                    auto tmp5 = out_ptr0[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp8 = out_ptr1[static_cast<long>((14L*(static_cast<long>((3L + x2 + (7L*x1))) % static_cast<long>(14L))) + (196L*x0) + (static_cast<long>((3L + x4 + (7L*x3))) % static_cast<long>(14L)))];
                                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x5));
                                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x5));
                                    auto tmp2 = tmp0 + tmp1;
                                    auto tmp4 = tmp2 + tmp3;
                                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                    auto tmp7 = tmp4 - tmp6;
                                    auto tmp9 = static_cast<float>(512.0);
                                    auto tmp10 = tmp8 / tmp9;
                                    auto tmp11 = static_cast<float>(1e-05);
                                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                    auto tmp13 = 1 / std::sqrt(tmp12);
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp7 * tmp14;
                                    auto tmp17 = tmp15 * tmp16;
                                    auto tmp19 = tmp17 + tmp18;
                                    tmp19.store(out_ptr2 + static_cast<long>(x5 + (512L*x4) + (3584L*x2) + (25088L*x3) + (50176L*x1) + (100352L*x0)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_130 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (75264L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (25088L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = in_ptr3[static_cast<long>(x3 + (49L*x2) + (2401L*(static_cast<long>(x0) % static_cast<long>(4L))))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (16L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (38416L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (25088L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*(static_cast<long>(x1) % static_cast<long>(14L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>(x1) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>(c10::div_floor_integer(((14L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(x1) % static_cast<long>(14L))), 14L)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*(static_cast<long>((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + (static_cast<long>(x1) % static_cast<long>(14L)))) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + (c10::div_floor_integer(x1, 14L)))) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(512.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_native_layer_norm_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (512L*(static_cast<long>(x2) % static_cast<long>(7L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer(x2, 7L))) + (50176L*(c10::div_floor_integer(x1, 7L))) + (100352L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (512L*(static_cast<long>((static_cast<long>((11L + x2)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (3584L*(static_cast<long>((static_cast<long>((11L + x1)) % static_cast<long>(14L))) % static_cast<long>(7L))) + (25088L*(c10::div_floor_integer((static_cast<long>((11L + x2)) % static_cast<long>(14L)), 7L))) + (50176L*(c10::div_floor_integer((static_cast<long>((11L + x1)) % static_cast<long>(14L)), 7L))) + (100352L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            tmp8.store(in_out_ptr0 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>((512L*(c10::div_floor_integer(x2, 1024L))) + (1024L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer(x2, 512L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>(x2) % static_cast<long>(512L)))];
                            tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                        }
                        out_ptr0[static_cast<long>(x1 + (7L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (7L*x0))] = tmp_acc0.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2048L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>((512L*(c10::div_floor_integer((x2 + x2_inner), 1024L))) + (1024L*x1) + (7168L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 512L)) % static_cast<long>(2L))) + (14336L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(512L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = out_ptr0[static_cast<long>(x1 + (7L*x0))];
                        auto tmp4 = out_ptr1[static_cast<long>(x1 + (7L*x0))];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(2048.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (2048L*x1) + (14336L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(1024.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (3072L*x2) + (150528L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(1024L + x1 + (3072L*x2) + (150528L*x0)), static_cast<long>(3072L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x1 + (3072L*x2) + (150528L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (76832L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (32L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (1568L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (76832L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (1568L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (32L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (76832L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(2048L + x3 + (32L*x1) + (3072L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (50176L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1024L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(1024.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(1024.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (3072L*x2) + (150528L*x0)));
                            auto tmp1 = static_cast<float>(0.1767766952966369);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(1024L + x1 + (3072L*x2) + (150528L*x0)), static_cast<long>(3072L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x1 + (3072L*x2) + (150528L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (50176L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (76832L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp2 = decltype(tmp1)(tmp1 + 169);
                                auto tmp3 = tmp1 < 0;
                                auto tmp4 = tmp3 ? tmp2 : tmp1;
                                TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                                auto tmp5 = in_ptr2[static_cast<long>(x1 + (32L*tmp4))];
                                auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (1568L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (76832L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (1568L*x0))];
                            auto tmp2 = decltype(tmp1)(tmp1 + 169);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 169L), "index out of bounds: 0 <= tmp4 < 169L")
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + (32L*tmp4))];
                            auto tmp6 = decltype(tmp0)(tmp0 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = std::exp(tmp8);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (76832L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(2048L + x3 + (32L*x1) + (3072L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1568L*x2) + (50176L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (1024L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*(static_cast<long>(x1) % static_cast<long>(7L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((7L*(c10::div_floor_integer(x1, 7L))) + (static_cast<long>(x1) % static_cast<long>(7L))), 7L)) % static_cast<long>(7L))) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*(static_cast<long>(x1) % static_cast<long>(7L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((7L*(c10::div_floor_integer(x1, 7L))) + (static_cast<long>(x1) % static_cast<long>(7L))), 7L)) % static_cast<long>(7L))) + (50176L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (49L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (49L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*(static_cast<long>(x1) % static_cast<long>(7L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((7L*(c10::div_floor_integer(x1, 7L))) + (static_cast<long>(x1) % static_cast<long>(7L))), 7L)) % static_cast<long>(7L))) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*(static_cast<long>(x1) % static_cast<long>(7L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((7L*(c10::div_floor_integer(x1, 7L))) + (static_cast<long>(x1) % static_cast<long>(7L))), 7L)) % static_cast<long>(7L))) + (50176L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (49L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(1024.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = static_cast<float>(0.7071067811865476);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp0 * tmp5;
                auto tmp7 = tmp6.erf();
                auto tmp8 = static_cast<float>(1.0);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp3 * tmp10;
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_mean_native_layer_norm_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*(static_cast<long>(x1) % static_cast<long>(7L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((7L*(c10::div_floor_integer(x1, 7L))) + (static_cast<long>(x1) % static_cast<long>(7L))), 7L)) % static_cast<long>(7L))) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*(static_cast<long>(x1) % static_cast<long>(7L))) + (7168L*(static_cast<long>(c10::div_floor_integer(((7L*(c10::div_floor_integer(x1, 7L))) + (static_cast<long>(x1) % static_cast<long>(7L))), 7L)) % static_cast<long>(7L))) + (50176L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (50176L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (49L*x0))];
                            auto tmp4 = out_ptr1[static_cast<long>(x2 + (49L*x0))];
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 - tmp2;
                            auto tmp5 = static_cast<float>(1024.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = 1 / std::sqrt(tmp8);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp3 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp15;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1 = args
    args.clear()
    assert_size_stride(arg0_1, (169, 4), (4, 1))
    assert_size_stride(arg1_1, (169, 4), (4, 1))
    assert_size_stride(arg2_1, (169, 8), (8, 1))
    assert_size_stride(arg3_1, (169, 8), (8, 1))
    assert_size_stride(arg4_1, (169, 16), (16, 1))
    assert_size_stride(arg5_1, (169, 16), (16, 1))
    assert_size_stride(arg6_1, (169, 16), (16, 1))
    assert_size_stride(arg7_1, (169, 16), (16, 1))
    assert_size_stride(arg8_1, (169, 16), (16, 1))
    assert_size_stride(arg9_1, (169, 16), (16, 1))
    assert_size_stride(arg10_1, (169, 16), (16, 1))
    assert_size_stride(arg11_1, (169, 16), (16, 1))
    assert_size_stride(arg12_1, (169, 16), (16, 1))
    assert_size_stride(arg13_1, (169, 16), (16, 1))
    assert_size_stride(arg14_1, (169, 16), (16, 1))
    assert_size_stride(arg15_1, (169, 16), (16, 1))
    assert_size_stride(arg16_1, (169, 16), (16, 1))
    assert_size_stride(arg17_1, (169, 16), (16, 1))
    assert_size_stride(arg18_1, (169, 16), (16, 1))
    assert_size_stride(arg19_1, (169, 16), (16, 1))
    assert_size_stride(arg20_1, (169, 16), (16, 1))
    assert_size_stride(arg21_1, (169, 16), (16, 1))
    assert_size_stride(arg22_1, (169, 32), (32, 1))
    assert_size_stride(arg23_1, (169, 32), (32, 1))
    assert_size_stride(arg24_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (384, 128), (128, 1))
    assert_size_stride(arg31_1, (384, ), (1, ))
    assert_size_stride(arg32_1, (128, 128), (128, 1))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (512, 128), (128, 1))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (128, 512), (512, 1))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (384, 128), (128, 1))
    assert_size_stride(arg43_1, (384, ), (1, ))
    assert_size_stride(arg44_1, (128, 128), (128, 1))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (512, 128), (128, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (128, 512), (512, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (256, 512), (512, 1))
    assert_size_stride(arg55_1, (256, ), (1, ))
    assert_size_stride(arg56_1, (256, ), (1, ))
    assert_size_stride(arg57_1, (768, 256), (256, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (256, 256), (256, 1))
    assert_size_stride(arg60_1, (256, ), (1, ))
    assert_size_stride(arg61_1, (256, ), (1, ))
    assert_size_stride(arg62_1, (256, ), (1, ))
    assert_size_stride(arg63_1, (1024, 256), (256, 1))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (256, 1024), (1024, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (256, ), (1, ))
    assert_size_stride(arg69_1, (768, 256), (256, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (256, 256), (256, 1))
    assert_size_stride(arg72_1, (256, ), (1, ))
    assert_size_stride(arg73_1, (256, ), (1, ))
    assert_size_stride(arg74_1, (256, ), (1, ))
    assert_size_stride(arg75_1, (1024, 256), (256, 1))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (256, 1024), (1024, 1))
    assert_size_stride(arg78_1, (256, ), (1, ))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (512, 1024), (1024, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (1536, 512), (512, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (512, 512), (512, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (2048, 512), (512, 1))
    assert_size_stride(arg91_1, (2048, ), (1, ))
    assert_size_stride(arg92_1, (512, 2048), (2048, 1))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (1536, 512), (512, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (512, 512), (512, 1))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (2048, 512), (512, 1))
    assert_size_stride(arg103_1, (2048, ), (1, ))
    assert_size_stride(arg104_1, (512, 2048), (2048, 1))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (512, ), (1, ))
    assert_size_stride(arg107_1, (512, ), (1, ))
    assert_size_stride(arg108_1, (1536, 512), (512, 1))
    assert_size_stride(arg109_1, (1536, ), (1, ))
    assert_size_stride(arg110_1, (512, 512), (512, 1))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (512, ), (1, ))
    assert_size_stride(arg114_1, (2048, 512), (512, 1))
    assert_size_stride(arg115_1, (2048, ), (1, ))
    assert_size_stride(arg116_1, (512, 2048), (2048, 1))
    assert_size_stride(arg117_1, (512, ), (1, ))
    assert_size_stride(arg118_1, (512, ), (1, ))
    assert_size_stride(arg119_1, (512, ), (1, ))
    assert_size_stride(arg120_1, (1536, 512), (512, 1))
    assert_size_stride(arg121_1, (1536, ), (1, ))
    assert_size_stride(arg122_1, (512, 512), (512, 1))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (2048, 512), (512, 1))
    assert_size_stride(arg127_1, (2048, ), (1, ))
    assert_size_stride(arg128_1, (512, 2048), (2048, 1))
    assert_size_stride(arg129_1, (512, ), (1, ))
    assert_size_stride(arg130_1, (512, ), (1, ))
    assert_size_stride(arg131_1, (512, ), (1, ))
    assert_size_stride(arg132_1, (1536, 512), (512, 1))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (512, 512), (512, 1))
    assert_size_stride(arg135_1, (512, ), (1, ))
    assert_size_stride(arg136_1, (512, ), (1, ))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (2048, 512), (512, 1))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (512, 2048), (2048, 1))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (512, ), (1, ))
    assert_size_stride(arg144_1, (1536, 512), (512, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (512, 512), (512, 1))
    assert_size_stride(arg147_1, (512, ), (1, ))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (2048, 512), (512, 1))
    assert_size_stride(arg151_1, (2048, ), (1, ))
    assert_size_stride(arg152_1, (512, 2048), (2048, 1))
    assert_size_stride(arg153_1, (512, ), (1, ))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (512, ), (1, ))
    assert_size_stride(arg156_1, (1536, 512), (512, 1))
    assert_size_stride(arg157_1, (1536, ), (1, ))
    assert_size_stride(arg158_1, (512, 512), (512, 1))
    assert_size_stride(arg159_1, (512, ), (1, ))
    assert_size_stride(arg160_1, (512, ), (1, ))
    assert_size_stride(arg161_1, (512, ), (1, ))
    assert_size_stride(arg162_1, (2048, 512), (512, 1))
    assert_size_stride(arg163_1, (2048, ), (1, ))
    assert_size_stride(arg164_1, (512, 2048), (2048, 1))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (512, ), (1, ))
    assert_size_stride(arg168_1, (1536, 512), (512, 1))
    assert_size_stride(arg169_1, (1536, ), (1, ))
    assert_size_stride(arg170_1, (512, 512), (512, 1))
    assert_size_stride(arg171_1, (512, ), (1, ))
    assert_size_stride(arg172_1, (512, ), (1, ))
    assert_size_stride(arg173_1, (512, ), (1, ))
    assert_size_stride(arg174_1, (2048, 512), (512, 1))
    assert_size_stride(arg175_1, (2048, ), (1, ))
    assert_size_stride(arg176_1, (512, 2048), (2048, 1))
    assert_size_stride(arg177_1, (512, ), (1, ))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (512, ), (1, ))
    assert_size_stride(arg180_1, (1536, 512), (512, 1))
    assert_size_stride(arg181_1, (1536, ), (1, ))
    assert_size_stride(arg182_1, (512, 512), (512, 1))
    assert_size_stride(arg183_1, (512, ), (1, ))
    assert_size_stride(arg184_1, (512, ), (1, ))
    assert_size_stride(arg185_1, (512, ), (1, ))
    assert_size_stride(arg186_1, (2048, 512), (512, 1))
    assert_size_stride(arg187_1, (2048, ), (1, ))
    assert_size_stride(arg188_1, (512, 2048), (2048, 1))
    assert_size_stride(arg189_1, (512, ), (1, ))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (512, ), (1, ))
    assert_size_stride(arg192_1, (1536, 512), (512, 1))
    assert_size_stride(arg193_1, (1536, ), (1, ))
    assert_size_stride(arg194_1, (512, 512), (512, 1))
    assert_size_stride(arg195_1, (512, ), (1, ))
    assert_size_stride(arg196_1, (512, ), (1, ))
    assert_size_stride(arg197_1, (512, ), (1, ))
    assert_size_stride(arg198_1, (2048, 512), (512, 1))
    assert_size_stride(arg199_1, (2048, ), (1, ))
    assert_size_stride(arg200_1, (512, 2048), (2048, 1))
    assert_size_stride(arg201_1, (512, ), (1, ))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (512, ), (1, ))
    assert_size_stride(arg204_1, (1536, 512), (512, 1))
    assert_size_stride(arg205_1, (1536, ), (1, ))
    assert_size_stride(arg206_1, (512, 512), (512, 1))
    assert_size_stride(arg207_1, (512, ), (1, ))
    assert_size_stride(arg208_1, (512, ), (1, ))
    assert_size_stride(arg209_1, (512, ), (1, ))
    assert_size_stride(arg210_1, (2048, 512), (512, 1))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (512, 2048), (2048, 1))
    assert_size_stride(arg213_1, (512, ), (1, ))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (512, ), (1, ))
    assert_size_stride(arg216_1, (1536, 512), (512, 1))
    assert_size_stride(arg217_1, (1536, ), (1, ))
    assert_size_stride(arg218_1, (512, 512), (512, 1))
    assert_size_stride(arg219_1, (512, ), (1, ))
    assert_size_stride(arg220_1, (512, ), (1, ))
    assert_size_stride(arg221_1, (512, ), (1, ))
    assert_size_stride(arg222_1, (2048, 512), (512, 1))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (512, 2048), (2048, 1))
    assert_size_stride(arg225_1, (512, ), (1, ))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (512, ), (1, ))
    assert_size_stride(arg228_1, (1536, 512), (512, 1))
    assert_size_stride(arg229_1, (1536, ), (1, ))
    assert_size_stride(arg230_1, (512, 512), (512, 1))
    assert_size_stride(arg231_1, (512, ), (1, ))
    assert_size_stride(arg232_1, (512, ), (1, ))
    assert_size_stride(arg233_1, (512, ), (1, ))
    assert_size_stride(arg234_1, (2048, 512), (512, 1))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (512, 2048), (2048, 1))
    assert_size_stride(arg237_1, (512, ), (1, ))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (512, ), (1, ))
    assert_size_stride(arg240_1, (1536, 512), (512, 1))
    assert_size_stride(arg241_1, (1536, ), (1, ))
    assert_size_stride(arg242_1, (512, 512), (512, 1))
    assert_size_stride(arg243_1, (512, ), (1, ))
    assert_size_stride(arg244_1, (512, ), (1, ))
    assert_size_stride(arg245_1, (512, ), (1, ))
    assert_size_stride(arg246_1, (2048, 512), (512, 1))
    assert_size_stride(arg247_1, (2048, ), (1, ))
    assert_size_stride(arg248_1, (512, 2048), (2048, 1))
    assert_size_stride(arg249_1, (512, ), (1, ))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (512, ), (1, ))
    assert_size_stride(arg252_1, (1536, 512), (512, 1))
    assert_size_stride(arg253_1, (1536, ), (1, ))
    assert_size_stride(arg254_1, (512, 512), (512, 1))
    assert_size_stride(arg255_1, (512, ), (1, ))
    assert_size_stride(arg256_1, (512, ), (1, ))
    assert_size_stride(arg257_1, (512, ), (1, ))
    assert_size_stride(arg258_1, (2048, 512), (512, 1))
    assert_size_stride(arg259_1, (2048, ), (1, ))
    assert_size_stride(arg260_1, (512, 2048), (2048, 1))
    assert_size_stride(arg261_1, (512, ), (1, ))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (512, ), (1, ))
    assert_size_stride(arg264_1, (1536, 512), (512, 1))
    assert_size_stride(arg265_1, (1536, ), (1, ))
    assert_size_stride(arg266_1, (512, 512), (512, 1))
    assert_size_stride(arg267_1, (512, ), (1, ))
    assert_size_stride(arg268_1, (512, ), (1, ))
    assert_size_stride(arg269_1, (512, ), (1, ))
    assert_size_stride(arg270_1, (2048, 512), (512, 1))
    assert_size_stride(arg271_1, (2048, ), (1, ))
    assert_size_stride(arg272_1, (512, 2048), (2048, 1))
    assert_size_stride(arg273_1, (512, ), (1, ))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (512, ), (1, ))
    assert_size_stride(arg276_1, (1536, 512), (512, 1))
    assert_size_stride(arg277_1, (1536, ), (1, ))
    assert_size_stride(arg278_1, (512, 512), (512, 1))
    assert_size_stride(arg279_1, (512, ), (1, ))
    assert_size_stride(arg280_1, (512, ), (1, ))
    assert_size_stride(arg281_1, (512, ), (1, ))
    assert_size_stride(arg282_1, (2048, 512), (512, 1))
    assert_size_stride(arg283_1, (2048, ), (1, ))
    assert_size_stride(arg284_1, (512, 2048), (2048, 1))
    assert_size_stride(arg285_1, (512, ), (1, ))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (512, ), (1, ))
    assert_size_stride(arg288_1, (1536, 512), (512, 1))
    assert_size_stride(arg289_1, (1536, ), (1, ))
    assert_size_stride(arg290_1, (512, 512), (512, 1))
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (512, ), (1, ))
    assert_size_stride(arg294_1, (2048, 512), (512, 1))
    assert_size_stride(arg295_1, (2048, ), (1, ))
    assert_size_stride(arg296_1, (512, 2048), (2048, 1))
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (2048, ), (1, ))
    assert_size_stride(arg299_1, (2048, ), (1, ))
    assert_size_stride(arg300_1, (1024, 2048), (2048, 1))
    assert_size_stride(arg301_1, (1024, ), (1, ))
    assert_size_stride(arg302_1, (1024, ), (1, ))
    assert_size_stride(arg303_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg304_1, (3072, ), (1, ))
    assert_size_stride(arg305_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg306_1, (1024, ), (1, ))
    assert_size_stride(arg307_1, (1024, ), (1, ))
    assert_size_stride(arg308_1, (1024, ), (1, ))
    assert_size_stride(arg309_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg310_1, (4096, ), (1, ))
    assert_size_stride(arg311_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg312_1, (1024, ), (1, ))
    assert_size_stride(arg313_1, (1024, ), (1, ))
    assert_size_stride(arg314_1, (1024, ), (1, ))
    assert_size_stride(arg315_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg316_1, (3072, ), (1, ))
    assert_size_stride(arg317_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg318_1, (1024, ), (1, ))
    assert_size_stride(arg319_1, (1024, ), (1, ))
    assert_size_stride(arg320_1, (1024, ), (1, ))
    assert_size_stride(arg321_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg322_1, (4096, ), (1, ))
    assert_size_stride(arg323_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg324_1, (1024, ), (1, ))
    assert_size_stride(arg325_1, (1024, ), (1, ))
    assert_size_stride(arg326_1, (1024, ), (1, ))
    assert_size_stride(arg327_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg328_1, (1000, ), (1, ))
    assert_size_stride(arg329_1, (49, 49), (49, 1))
    assert_size_stride(arg330_1, (64, 49, 49), (2401, 49, 1))
    assert_size_stride(arg331_1, (49, 49), (49, 1))
    assert_size_stride(arg332_1, (49, 49), (49, 1))
    assert_size_stride(arg333_1, (16, 49, 49), (2401, 49, 1))
    assert_size_stride(arg334_1, (49, 49), (49, 1))
    assert_size_stride(arg335_1, (49, 49), (49, 1))
    assert_size_stride(arg336_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg337_1, (49, 49), (49, 1))
    assert_size_stride(arg338_1, (49, 49), (49, 1))
    assert_size_stride(arg339_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg340_1, (49, 49), (49, 1))
    assert_size_stride(arg341_1, (49, 49), (49, 1))
    assert_size_stride(arg342_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg343_1, (49, 49), (49, 1))
    assert_size_stride(arg344_1, (49, 49), (49, 1))
    assert_size_stride(arg345_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg346_1, (49, 49), (49, 1))
    assert_size_stride(arg347_1, (49, 49), (49, 1))
    assert_size_stride(arg348_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg349_1, (49, 49), (49, 1))
    assert_size_stride(arg350_1, (49, 49), (49, 1))
    assert_size_stride(arg351_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg352_1, (49, 49), (49, 1))
    assert_size_stride(arg353_1, (49, 49), (49, 1))
    assert_size_stride(arg354_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg355_1, (49, 49), (49, 1))
    assert_size_stride(arg356_1, (49, 49), (49, 1))
    assert_size_stride(arg357_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg358_1, (49, 49), (49, 1))
    assert_size_stride(arg359_1, (49, 49), (49, 1))
    assert_size_stride(arg360_1, (4, 49, 49), (2401, 49, 1))
    assert_size_stride(arg361_1, (49, 49), (49, 1))
    assert_size_stride(arg362_1, (49, 49), (49, 1))
    assert_size_stride(arg363_1, (49, 49), (49, 1))
    assert_size_stride(arg364_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg364_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg24_1
    del arg364_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg25_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del arg25_1
    del buf1
    buf3 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 56, 56, 128), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf10 = empty((8, 8, 8, 7, 7, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg26_1
    del arg27_1
    del arg28_1
    del arg29_1
    del buf3
    del buf4
    buf11 = empty((25088, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf10, (25088, 128), (128, 1), 0), reinterpret_tensor(arg30_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf11)
    del arg30_1
    del arg31_1
    buf12 = reinterpret_tensor(buf10, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf10  # reuse
    buf13 = reinterpret_tensor(buf2, (512, 4, 32, 49), (6272, 1568, 49, 1), 0); del buf2  # reuse
    cpp_fused_clone_mul_2(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    buf14 = empty((2048, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf12, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf13, (2048, 32, 49), (1568, 49, 1), 0), out=buf14)
    buf15 = empty_strided((512, 4, 49, 1), (196, 49, 1, 100352), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf14, (512, 4, 49, 49), (9604, 2401, 49, 1), 0); del buf14  # reuse
    buf17 = empty_strided((512, 4, 49, 1), (196, 49, 1, 100352), device='cpu', dtype=torch.float32)
    buf18 = buf16; del buf16  # reuse
    buf19 = reinterpret_tensor(buf13, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf13  # reuse
    cpp_fused__softmax_add_clone_3(c_void_p(buf18.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()))
    del arg0_1
    del arg329_1
    buf20 = reinterpret_tensor(buf12, (2048, 49, 32), (1568, 32, 1), 0); del buf12  # reuse
    # Source Nodes: [x_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf18, (2048, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf19, (2048, 49, 32), (1568, 32, 1), 0), out=buf20)
    buf21 = reinterpret_tensor(buf19, (512, 49, 4, 32), (6272, 128, 32, 1), 0); del buf19  # reuse
    cpp_fused_clone_4(c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = reinterpret_tensor(buf20, (25088, 128), (128, 1), 0); del buf20  # reuse
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg33_1, reinterpret_tensor(buf21, (25088, 128), (128, 1), 0), reinterpret_tensor(arg32_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf22)
    del arg32_1
    del arg33_1
    buf23 = reinterpret_tensor(buf8, (8, 3136, 1), (3136, 1, 25088), 0); del buf8  # reuse
    buf24 = reinterpret_tensor(buf7, (8, 3136, 1), (3136, 1, 25088), 0); del buf7  # reuse
    buf26 = reinterpret_tensor(buf21, (8, 3136, 128), (401408, 128, 1), 0); del buf21  # reuse
    cpp_fused_native_layer_norm_5(c_void_p(buf6.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()))
    del arg34_1
    del arg35_1
    buf27 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf26, (25088, 128), (128, 1), 0), reinterpret_tensor(arg36_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf27)
    del arg36_1
    del arg37_1
    buf28 = reinterpret_tensor(buf27, (8, 3136, 512), (1605632, 512, 1), 0); del buf27  # reuse
    cpp_fused_gelu_6(c_void_p(buf28.data_ptr()))
    buf29 = reinterpret_tensor(buf26, (25088, 128), (128, 1), 0); del buf26  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg39_1, reinterpret_tensor(buf28, (25088, 512), (512, 1), 0), reinterpret_tensor(arg38_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf29)
    del arg38_1
    del arg39_1
    buf30 = reinterpret_tensor(buf24, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf24  # reuse
    buf31 = reinterpret_tensor(buf23, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf23  # reuse
    buf33 = empty((8, 8, 8, 7, 7, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_7(c_void_p(buf6.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg40_1
    del arg41_1
    buf34 = buf11; del buf11  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___0___blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, reinterpret_tensor(buf33, (25088, 128), (128, 1), 0), reinterpret_tensor(arg42_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf34)
    del arg42_1
    del arg43_1
    buf35 = reinterpret_tensor(buf33, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf33  # reuse
    buf36 = empty((512, 4, 32, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_8(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    buf37 = reinterpret_tensor(buf18, (2048, 49, 49), (2401, 49, 1), 0); del buf18  # reuse
    # Source Nodes: [attn_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf35, (2048, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf36, (2048, 32, 49), (1568, 49, 1), 0), out=buf37)
    buf38 = buf17; del buf17  # reuse
    buf39 = reinterpret_tensor(buf37, (512, 4, 49, 49), (9604, 2401, 49, 1), 0); del buf37  # reuse
    buf40 = buf15; del buf15  # reuse
    buf41 = buf39; del buf39  # reuse
    buf42 = reinterpret_tensor(buf36, (512, 4, 49, 32), (6272, 1568, 32, 1), 0); del buf36  # reuse
    cpp_fused__softmax_clone_9(c_void_p(buf41.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg1_1
    del arg330_1
    del arg331_1
    del buf34
    del buf38
    del buf40
    buf43 = reinterpret_tensor(buf35, (2048, 49, 32), (1568, 32, 1), 0); del buf35  # reuse
    # Source Nodes: [x_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf41, (2048, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf42, (2048, 49, 32), (1568, 32, 1), 0), out=buf43)
    del buf41
    buf44 = reinterpret_tensor(buf42, (512, 49, 4, 32), (6272, 128, 32, 1), 0); del buf42  # reuse
    cpp_fused_clone_10(c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = reinterpret_tensor(buf43, (25088, 128), (128, 1), 0); del buf43  # reuse
    # Source Nodes: [x_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg45_1, reinterpret_tensor(buf44, (25088, 128), (128, 1), 0), reinterpret_tensor(arg44_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf45)
    del arg44_1
    del arg45_1
    buf46 = reinterpret_tensor(buf31, (8, 3136, 1), (3136, 1, 25088), 0); del buf31  # reuse
    buf47 = reinterpret_tensor(buf30, (8, 3136, 1), (3136, 1, 25088), 0); del buf30  # reuse
    buf49 = reinterpret_tensor(buf44, (8, 3136, 128), (401408, 128, 1), 0); del buf44  # reuse
    cpp_fused_native_layer_norm_11(c_void_p(buf6.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg46_1
    del arg47_1
    buf50 = reinterpret_tensor(buf28, (25088, 512), (512, 1), 0); del buf28  # reuse
    # Source Nodes: [x_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf49, (25088, 128), (128, 1), 0), reinterpret_tensor(arg48_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf50)
    del arg48_1
    del arg49_1
    buf51 = reinterpret_tensor(buf50, (8, 3136, 512), (1605632, 512, 1), 0); del buf50  # reuse
    cpp_fused_gelu_12(c_void_p(buf51.data_ptr()))
    buf52 = reinterpret_tensor(buf49, (25088, 128), (128, 1), 0); del buf49  # reuse
    # Source Nodes: [x_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg51_1, reinterpret_tensor(buf51, (25088, 512), (512, 1), 0), reinterpret_tensor(arg50_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf52)
    del arg50_1
    del arg51_1
    del buf51
    buf53 = reinterpret_tensor(buf52, (8, 28, 28, 2, 2, 128), (401408, 14336, 256, 128, 7168, 1), 0); del buf52  # reuse
    buf54 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf57 = empty((8, 28, 28, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_13(c_void_p(buf53.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg52_1
    del arg53_1
    del buf22
    del buf29
    del buf45
    del buf53
    del buf6
    buf58 = empty((6272, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf57, (6272, 512), (512, 1), 0), reinterpret_tensor(arg54_1, (512, 256), (1, 512), 0), out=buf58)
    del arg54_1
    buf59 = buf55; del buf55  # reuse
    buf60 = buf54; del buf54  # reuse
    buf62 = empty((8, 4, 4, 7, 7, 256), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_14(c_void_p(buf58.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf62.data_ptr()))
    del arg55_1
    del arg56_1
    buf63 = empty((6272, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf62, (6272, 256), (256, 1), 0), reinterpret_tensor(arg57_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf63)
    del arg57_1
    del arg58_1
    buf64 = reinterpret_tensor(buf62, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf62  # reuse
    buf65 = empty((128, 8, 32, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_15(c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    buf66 = empty((1024, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf64, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf65, (1024, 32, 49), (1568, 49, 1), 0), out=buf66)
    buf67 = empty_strided((128, 8, 49, 1), (392, 49, 1, 50176), device='cpu', dtype=torch.float32)
    buf68 = reinterpret_tensor(buf66, (128, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf66  # reuse
    buf69 = empty_strided((128, 8, 49, 1), (392, 49, 1, 50176), device='cpu', dtype=torch.float32)
    buf70 = buf68; del buf68  # reuse
    buf71 = reinterpret_tensor(buf65, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf65  # reuse
    cpp_fused__softmax_add_clone_16(c_void_p(buf70.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()))
    del arg2_1
    del arg332_1
    buf72 = reinterpret_tensor(buf64, (1024, 49, 32), (1568, 32, 1), 0); del buf64  # reuse
    # Source Nodes: [x_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf70, (1024, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf71, (1024, 49, 32), (1568, 32, 1), 0), out=buf72)
    buf73 = reinterpret_tensor(buf71, (128, 49, 8, 32), (12544, 256, 32, 1), 0); del buf71  # reuse
    cpp_fused_clone_17(c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    buf74 = reinterpret_tensor(buf72, (6272, 256), (256, 1), 0); del buf72  # reuse
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf73, (6272, 256), (256, 1), 0), reinterpret_tensor(arg59_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf74)
    del arg59_1
    del arg60_1
    buf75 = reinterpret_tensor(buf60, (8, 784, 1), (784, 1, 6272), 0); del buf60  # reuse
    buf76 = reinterpret_tensor(buf59, (8, 784, 1), (784, 1, 6272), 0); del buf59  # reuse
    buf78 = reinterpret_tensor(buf73, (8, 784, 256), (200704, 256, 1), 0); del buf73  # reuse
    cpp_fused_native_layer_norm_18(c_void_p(buf58.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg61_1
    del arg62_1
    buf79 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf78, (6272, 256), (256, 1), 0), reinterpret_tensor(arg63_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf79)
    del arg63_1
    del arg64_1
    buf80 = reinterpret_tensor(buf79, (8, 784, 1024), (802816, 1024, 1), 0); del buf79  # reuse
    cpp_fused_gelu_19(c_void_p(buf80.data_ptr()))
    buf81 = reinterpret_tensor(buf78, (6272, 256), (256, 1), 0); del buf78  # reuse
    # Source Nodes: [x_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf80, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg65_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf81)
    del arg65_1
    del arg66_1
    buf82 = reinterpret_tensor(buf76, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf76  # reuse
    buf83 = reinterpret_tensor(buf75, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf75  # reuse
    buf85 = empty((8, 4, 4, 7, 7, 256), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_20(c_void_p(buf58.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg67_1
    del arg68_1
    buf86 = buf63; del buf63  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___1___blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf85, (6272, 256), (256, 1), 0), reinterpret_tensor(arg69_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf86)
    del arg69_1
    del arg70_1
    buf87 = reinterpret_tensor(buf85, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf85  # reuse
    buf88 = empty((128, 8, 32, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_21(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    buf89 = reinterpret_tensor(buf70, (1024, 49, 49), (2401, 49, 1), 0); del buf70  # reuse
    # Source Nodes: [attn_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf87, (1024, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf88, (1024, 32, 49), (1568, 49, 1), 0), out=buf89)
    buf90 = buf69; del buf69  # reuse
    buf91 = reinterpret_tensor(buf89, (128, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf89  # reuse
    buf92 = buf67; del buf67  # reuse
    buf93 = buf91; del buf91  # reuse
    buf94 = reinterpret_tensor(buf88, (128, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf88  # reuse
    cpp_fused__softmax_clone_22(c_void_p(buf93.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()))
    del arg333_1
    del arg334_1
    del arg3_1
    del buf86
    del buf90
    del buf92
    buf95 = reinterpret_tensor(buf87, (1024, 49, 32), (1568, 32, 1), 0); del buf87  # reuse
    # Source Nodes: [x_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf93, (1024, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf94, (1024, 49, 32), (1568, 32, 1), 0), out=buf95)
    del buf93
    buf96 = reinterpret_tensor(buf94, (128, 49, 8, 32), (12544, 256, 32, 1), 0); del buf94  # reuse
    cpp_fused_clone_23(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    buf97 = reinterpret_tensor(buf95, (6272, 256), (256, 1), 0); del buf95  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf96, (6272, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf97)
    del arg71_1
    del arg72_1
    buf98 = reinterpret_tensor(buf83, (8, 784, 1), (784, 1, 6272), 0); del buf83  # reuse
    buf99 = reinterpret_tensor(buf82, (8, 784, 1), (784, 1, 6272), 0); del buf82  # reuse
    buf101 = reinterpret_tensor(buf96, (8, 784, 256), (200704, 256, 1), 0); del buf96  # reuse
    cpp_fused_native_layer_norm_24(c_void_p(buf58.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg73_1
    del arg74_1
    del buf98
    del buf99
    buf102 = reinterpret_tensor(buf80, (6272, 1024), (1024, 1), 0); del buf80  # reuse
    # Source Nodes: [x_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf101, (6272, 256), (256, 1), 0), reinterpret_tensor(arg75_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf102)
    del arg75_1
    del arg76_1
    buf103 = reinterpret_tensor(buf102, (8, 784, 1024), (802816, 1024, 1), 0); del buf102  # reuse
    cpp_fused_gelu_25(c_void_p(buf103.data_ptr()))
    buf104 = reinterpret_tensor(buf101, (6272, 256), (256, 1), 0); del buf101  # reuse
    # Source Nodes: [x_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf103, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg77_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf104)
    del arg77_1
    del arg78_1
    del buf103
    buf105 = reinterpret_tensor(buf104, (8, 14, 14, 2, 2, 256), (200704, 14336, 512, 256, 7168, 1), 0); del buf104  # reuse
    buf106 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf107 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf109 = empty((8, 14, 14, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_26(c_void_p(buf105.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg79_1
    del arg80_1
    del buf105
    del buf58
    del buf74
    del buf81
    del buf97
    buf110 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_87], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf109, (1568, 1024), (1024, 1), 0), reinterpret_tensor(arg81_1, (1024, 512), (1, 1024), 0), out=buf110)
    del arg81_1
    buf111 = buf107; del buf107  # reuse
    buf112 = buf106; del buf106  # reuse
    buf114 = empty((8, 2, 2, 7, 7, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_27(c_void_p(buf110.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()))
    del arg82_1
    del arg83_1
    buf115 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf114, (1568, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf115)
    del arg84_1
    del arg85_1
    buf116 = reinterpret_tensor(buf114, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf114  # reuse
    buf117 = empty((32, 16, 32, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_28(c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = empty((512, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf116, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf117, (512, 32, 49), (1568, 49, 1), 0), out=buf118)
    buf119 = reinterpret_tensor(buf47, (32, 16, 49, 1), (784, 49, 1, 25088), 0); del buf47  # reuse
    buf120 = reinterpret_tensor(buf118, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf118  # reuse
    buf121 = reinterpret_tensor(buf46, (32, 16, 49, 1), (784, 49, 1, 25088), 0); del buf46  # reuse
    buf122 = buf120; del buf120  # reuse
    buf123 = reinterpret_tensor(buf117, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf117  # reuse
    cpp_fused__softmax_add_clone_29(c_void_p(buf122.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg335_1
    del arg4_1
    buf124 = reinterpret_tensor(buf116, (512, 49, 32), (1568, 32, 1), 0); del buf116  # reuse
    # Source Nodes: [x_89], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf122, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf123, (512, 49, 32), (1568, 32, 1), 0), out=buf124)
    buf125 = reinterpret_tensor(buf123, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf123  # reuse
    cpp_fused_clone_30(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf124, (1568, 512), (512, 1), 0); del buf124  # reuse
    # Source Nodes: [x_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf125, (1568, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf126)
    del arg86_1
    del arg87_1
    buf127 = reinterpret_tensor(buf112, (8, 196, 1), (196, 1, 1568), 0); del buf112  # reuse
    buf128 = reinterpret_tensor(buf111, (8, 196, 1), (196, 1, 1568), 0); del buf111  # reuse
    buf130 = reinterpret_tensor(buf125, (8, 196, 512), (100352, 512, 1), 0); del buf125  # reuse
    cpp_fused_native_layer_norm_31(c_void_p(buf110.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg88_1
    del arg89_1
    buf131 = reinterpret_tensor(buf57, (1568, 2048), (2048, 1), 0); del buf57  # reuse
    # Source Nodes: [x_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf130, (1568, 512), (512, 1), 0), reinterpret_tensor(arg90_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf131)
    del arg90_1
    del arg91_1
    buf132 = reinterpret_tensor(buf131, (8, 196, 2048), (401408, 2048, 1), 0); del buf131  # reuse
    cpp_fused_gelu_32(c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf130, (1568, 512), (512, 1), 0); del buf130  # reuse
    # Source Nodes: [x_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf132, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg92_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf133)
    del arg92_1
    del arg93_1
    buf134 = reinterpret_tensor(buf128, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf128  # reuse
    buf135 = reinterpret_tensor(buf127, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf127  # reuse
    buf137 = empty((8, 2, 2, 7, 7, 512), device='cpu', dtype=torch.float32)
    cpp_fused_clone_native_layer_norm_33(c_void_p(buf110.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf137.data_ptr()))
    del arg94_1
    del arg95_1
    buf138 = buf115; del buf115  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf137, (1568, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf138)
    del arg96_1
    del arg97_1
    buf139 = reinterpret_tensor(buf137, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf137  # reuse
    buf140 = empty((32, 16, 32, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_34(c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    buf141 = reinterpret_tensor(buf122, (512, 49, 49), (2401, 49, 1), 0); del buf122  # reuse
    # Source Nodes: [attn_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf139, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf140, (512, 32, 49), (1568, 49, 1), 0), out=buf141)
    buf142 = buf121; del buf121  # reuse
    buf143 = reinterpret_tensor(buf141, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf141  # reuse
    buf144 = buf119; del buf119  # reuse
    buf145 = buf143; del buf143  # reuse
    buf146 = reinterpret_tensor(buf140, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf140  # reuse
    cpp_fused__softmax_clone_35(c_void_p(buf145.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()))
    del arg336_1
    del arg337_1
    del arg5_1
    buf147 = reinterpret_tensor(buf139, (512, 49, 32), (1568, 32, 1), 0); del buf139  # reuse
    # Source Nodes: [x_107], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf145, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf146, (512, 49, 32), (1568, 32, 1), 0), out=buf147)
    buf148 = reinterpret_tensor(buf146, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf146  # reuse
    cpp_fused_clone_36(c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    buf149 = reinterpret_tensor(buf147, (1568, 512), (512, 1), 0); del buf147  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf148, (1568, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf149)
    del arg98_1
    del arg99_1
    buf150 = reinterpret_tensor(buf135, (8, 196, 1), (196, 1, 1568), 0); del buf135  # reuse
    buf151 = reinterpret_tensor(buf134, (8, 196, 1), (196, 1, 1568), 0); del buf134  # reuse
    buf153 = reinterpret_tensor(buf148, (8, 196, 512), (100352, 512, 1), 0); del buf148  # reuse
    cpp_fused_native_layer_norm_37(c_void_p(buf110.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg100_1
    del arg101_1
    buf154 = reinterpret_tensor(buf132, (1568, 2048), (2048, 1), 0); del buf132  # reuse
    # Source Nodes: [x_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf153, (1568, 512), (512, 1), 0), reinterpret_tensor(arg102_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf154)
    del arg102_1
    del arg103_1
    buf155 = reinterpret_tensor(buf154, (8, 196, 2048), (401408, 2048, 1), 0); del buf154  # reuse
    cpp_fused_gelu_38(c_void_p(buf155.data_ptr()))
    buf156 = reinterpret_tensor(buf153, (1568, 512), (512, 1), 0); del buf153  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf155, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg104_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf156)
    del arg104_1
    del arg105_1
    buf157 = reinterpret_tensor(buf156, (8, 196, 512), (100352, 512, 1), 0); del buf156  # reuse
    buf158 = reinterpret_tensor(buf151, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf151  # reuse
    buf159 = reinterpret_tensor(buf150, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf150  # reuse
    buf161 = empty((8, 2, 2, 7, 7, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_39(c_void_p(buf157.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    del arg106_1
    del arg107_1
    buf162 = buf138; del buf138  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf161, (1568, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf162)
    del arg108_1
    del arg109_1
    buf163 = reinterpret_tensor(buf161, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf161  # reuse
    buf164 = reinterpret_tensor(buf149, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf149  # reuse
    cpp_fused_clone_mul_40(c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf145, (512, 49, 49), (2401, 49, 1), 0); del buf145  # reuse
    # Source Nodes: [attn_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf164, (512, 32, 49), (1568, 49, 1), 0), out=buf165)
    buf166 = buf144; del buf144  # reuse
    buf167 = reinterpret_tensor(buf165, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf165  # reuse
    buf168 = buf142; del buf142  # reuse
    buf169 = buf167; del buf167  # reuse
    buf170 = reinterpret_tensor(buf164, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf164  # reuse
    cpp_fused__softmax_add_clone_41(c_void_p(buf169.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()))
    del arg338_1
    del arg6_1
    buf171 = reinterpret_tensor(buf163, (512, 49, 32), (1568, 32, 1), 0); del buf163  # reuse
    # Source Nodes: [x_125], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf170, (512, 49, 32), (1568, 32, 1), 0), out=buf171)
    buf172 = reinterpret_tensor(buf170, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf170  # reuse
    cpp_fused_clone_42(c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    buf173 = reinterpret_tensor(buf171, (1568, 512), (512, 1), 0); del buf171  # reuse
    # Source Nodes: [x_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf172, (1568, 512), (512, 1), 0), reinterpret_tensor(arg110_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf173)
    del arg110_1
    del arg111_1
    buf174 = reinterpret_tensor(buf159, (8, 196, 1), (196, 1, 1568), 0); del buf159  # reuse
    buf175 = reinterpret_tensor(buf158, (8, 196, 1), (196, 1, 1568), 0); del buf158  # reuse
    buf177 = reinterpret_tensor(buf172, (8, 196, 512), (100352, 512, 1), 0); del buf172  # reuse
    cpp_fused_native_layer_norm_43(c_void_p(buf157.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg112_1
    del arg113_1
    buf178 = reinterpret_tensor(buf155, (1568, 2048), (2048, 1), 0); del buf155  # reuse
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf177, (1568, 512), (512, 1), 0), reinterpret_tensor(arg114_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf178)
    del arg114_1
    del arg115_1
    buf179 = reinterpret_tensor(buf178, (8, 196, 2048), (401408, 2048, 1), 0); del buf178  # reuse
    cpp_fused_gelu_44(c_void_p(buf179.data_ptr()))
    buf180 = reinterpret_tensor(buf177, (1568, 512), (512, 1), 0); del buf177  # reuse
    # Source Nodes: [x_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf179, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg116_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf180)
    del arg116_1
    del arg117_1
    buf181 = reinterpret_tensor(buf175, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf175  # reuse
    buf182 = reinterpret_tensor(buf174, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf174  # reuse
    buf184 = reinterpret_tensor(buf133, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf133  # reuse
    cpp_fused_clone_native_layer_norm_45(c_void_p(buf157.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg118_1
    del arg119_1
    buf185 = buf162; del buf162  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf184, (1568, 512), (512, 1), 0), reinterpret_tensor(arg120_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf185)
    del arg120_1
    del arg121_1
    buf186 = reinterpret_tensor(buf184, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf184  # reuse
    buf187 = reinterpret_tensor(buf126, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf126  # reuse
    cpp_fused_clone_mul_46(c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    buf188 = reinterpret_tensor(buf169, (512, 49, 49), (2401, 49, 1), 0); del buf169  # reuse
    # Source Nodes: [attn_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf186, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf187, (512, 32, 49), (1568, 49, 1), 0), out=buf188)
    buf189 = buf168; del buf168  # reuse
    buf190 = reinterpret_tensor(buf188, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf188  # reuse
    buf191 = buf166; del buf166  # reuse
    buf192 = buf190; del buf190  # reuse
    buf193 = reinterpret_tensor(buf187, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf187  # reuse
    cpp_fused__softmax_clone_47(c_void_p(buf192.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()))
    del arg339_1
    del arg340_1
    del arg7_1
    buf194 = reinterpret_tensor(buf186, (512, 49, 32), (1568, 32, 1), 0); del buf186  # reuse
    # Source Nodes: [x_143], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf192, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf193, (512, 49, 32), (1568, 32, 1), 0), out=buf194)
    buf195 = reinterpret_tensor(buf193, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf193  # reuse
    cpp_fused_clone_48(c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    buf196 = reinterpret_tensor(buf194, (1568, 512), (512, 1), 0); del buf194  # reuse
    # Source Nodes: [x_145], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf195, (1568, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf196)
    del arg122_1
    del arg123_1
    buf197 = reinterpret_tensor(buf182, (8, 196, 1), (196, 1, 1568), 0); del buf182  # reuse
    buf198 = reinterpret_tensor(buf181, (8, 196, 1), (196, 1, 1568), 0); del buf181  # reuse
    buf200 = reinterpret_tensor(buf195, (8, 196, 512), (100352, 512, 1), 0); del buf195  # reuse
    cpp_fused_native_layer_norm_49(c_void_p(buf157.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()))
    del arg124_1
    del arg125_1
    buf201 = reinterpret_tensor(buf179, (1568, 2048), (2048, 1), 0); del buf179  # reuse
    # Source Nodes: [x_152], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf200, (1568, 512), (512, 1), 0), reinterpret_tensor(arg126_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf201)
    del arg126_1
    del arg127_1
    buf202 = reinterpret_tensor(buf201, (8, 196, 2048), (401408, 2048, 1), 0); del buf201  # reuse
    cpp_fused_gelu_50(c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf200, (1568, 512), (512, 1), 0); del buf200  # reuse
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf202, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg128_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf203)
    del arg128_1
    del arg129_1
    buf204 = reinterpret_tensor(buf203, (8, 196, 512), (100352, 512, 1), 0); del buf203  # reuse
    buf205 = reinterpret_tensor(buf198, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf198  # reuse
    buf206 = reinterpret_tensor(buf197, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf197  # reuse
    buf208 = reinterpret_tensor(buf110, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf110  # reuse
    cpp_fused_add_clone_native_layer_norm_51(c_void_p(buf204.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()))
    del arg130_1
    del arg131_1
    buf209 = buf185; del buf185  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___4___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf208, (1568, 512), (512, 1), 0), reinterpret_tensor(arg132_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf209)
    del arg132_1
    del arg133_1
    buf210 = reinterpret_tensor(buf208, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf208  # reuse
    buf211 = reinterpret_tensor(buf196, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf196  # reuse
    cpp_fused_clone_mul_52(c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    buf212 = reinterpret_tensor(buf192, (512, 49, 49), (2401, 49, 1), 0); del buf192  # reuse
    # Source Nodes: [attn_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf210, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf211, (512, 32, 49), (1568, 49, 1), 0), out=buf212)
    buf213 = buf191; del buf191  # reuse
    buf214 = reinterpret_tensor(buf212, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf212  # reuse
    buf215 = buf189; del buf189  # reuse
    buf216 = buf214; del buf214  # reuse
    buf217 = reinterpret_tensor(buf211, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf211  # reuse
    cpp_fused__softmax_add_clone_53(c_void_p(buf216.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()))
    del arg341_1
    del arg8_1
    buf218 = reinterpret_tensor(buf210, (512, 49, 32), (1568, 32, 1), 0); del buf210  # reuse
    # Source Nodes: [x_161], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf216, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf217, (512, 49, 32), (1568, 32, 1), 0), out=buf218)
    buf219 = reinterpret_tensor(buf217, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf217  # reuse
    cpp_fused_clone_54(c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = reinterpret_tensor(buf218, (1568, 512), (512, 1), 0); del buf218  # reuse
    # Source Nodes: [x_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf219, (1568, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf220)
    del arg134_1
    del arg135_1
    buf221 = reinterpret_tensor(buf206, (8, 196, 1), (196, 1, 1568), 0); del buf206  # reuse
    buf222 = reinterpret_tensor(buf205, (8, 196, 1), (196, 1, 1568), 0); del buf205  # reuse
    buf224 = reinterpret_tensor(buf219, (8, 196, 512), (100352, 512, 1), 0); del buf219  # reuse
    cpp_fused_native_layer_norm_55(c_void_p(buf204.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()))
    del arg136_1
    del arg137_1
    buf225 = reinterpret_tensor(buf202, (1568, 2048), (2048, 1), 0); del buf202  # reuse
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf224, (1568, 512), (512, 1), 0), reinterpret_tensor(arg138_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf225)
    del arg138_1
    del arg139_1
    buf226 = reinterpret_tensor(buf225, (8, 196, 2048), (401408, 2048, 1), 0); del buf225  # reuse
    cpp_fused_gelu_56(c_void_p(buf226.data_ptr()))
    buf227 = reinterpret_tensor(buf224, (1568, 512), (512, 1), 0); del buf224  # reuse
    # Source Nodes: [x_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf226, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg140_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf227)
    del arg140_1
    del arg141_1
    buf228 = reinterpret_tensor(buf222, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf222  # reuse
    buf229 = reinterpret_tensor(buf221, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf221  # reuse
    buf231 = reinterpret_tensor(buf180, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf180  # reuse
    cpp_fused_clone_native_layer_norm_57(c_void_p(buf204.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()))
    del arg142_1
    del arg143_1
    buf232 = buf209; del buf209  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___5___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf231, (1568, 512), (512, 1), 0), reinterpret_tensor(arg144_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf232)
    del arg144_1
    del arg145_1
    buf233 = reinterpret_tensor(buf231, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf231  # reuse
    buf234 = reinterpret_tensor(buf173, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf173  # reuse
    cpp_fused_clone_mul_58(c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = reinterpret_tensor(buf216, (512, 49, 49), (2401, 49, 1), 0); del buf216  # reuse
    # Source Nodes: [attn_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf234, (512, 32, 49), (1568, 49, 1), 0), out=buf235)
    buf236 = buf215; del buf215  # reuse
    buf237 = reinterpret_tensor(buf235, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf235  # reuse
    buf238 = buf213; del buf213  # reuse
    buf239 = buf237; del buf237  # reuse
    buf240 = reinterpret_tensor(buf234, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf234  # reuse
    cpp_fused__softmax_clone_59(c_void_p(buf239.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg342_1
    del arg343_1
    del arg9_1
    buf241 = reinterpret_tensor(buf233, (512, 49, 32), (1568, 32, 1), 0); del buf233  # reuse
    # Source Nodes: [x_179], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf239, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf240, (512, 49, 32), (1568, 32, 1), 0), out=buf241)
    buf242 = reinterpret_tensor(buf240, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf240  # reuse
    cpp_fused_clone_60(c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf241, (1568, 512), (512, 1), 0); del buf241  # reuse
    # Source Nodes: [x_181], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg147_1, reinterpret_tensor(buf242, (1568, 512), (512, 1), 0), reinterpret_tensor(arg146_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf243)
    del arg146_1
    del arg147_1
    buf244 = reinterpret_tensor(buf229, (8, 196, 1), (196, 1, 1568), 0); del buf229  # reuse
    buf245 = reinterpret_tensor(buf228, (8, 196, 1), (196, 1, 1568), 0); del buf228  # reuse
    buf247 = reinterpret_tensor(buf242, (8, 196, 512), (100352, 512, 1), 0); del buf242  # reuse
    cpp_fused_native_layer_norm_61(c_void_p(buf204.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg148_1
    del arg149_1
    buf248 = reinterpret_tensor(buf226, (1568, 2048), (2048, 1), 0); del buf226  # reuse
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf247, (1568, 512), (512, 1), 0), reinterpret_tensor(arg150_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf248)
    del arg150_1
    del arg151_1
    buf249 = reinterpret_tensor(buf248, (8, 196, 2048), (401408, 2048, 1), 0); del buf248  # reuse
    cpp_fused_gelu_62(c_void_p(buf249.data_ptr()))
    buf250 = reinterpret_tensor(buf247, (1568, 512), (512, 1), 0); del buf247  # reuse
    # Source Nodes: [x_192], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf249, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg152_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf250)
    del arg152_1
    del arg153_1
    buf251 = reinterpret_tensor(buf250, (8, 196, 512), (100352, 512, 1), 0); del buf250  # reuse
    buf252 = reinterpret_tensor(buf245, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf245  # reuse
    buf253 = reinterpret_tensor(buf244, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf244  # reuse
    buf255 = reinterpret_tensor(buf157, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf157  # reuse
    cpp_fused_add_clone_native_layer_norm_63(c_void_p(buf251.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()))
    del arg154_1
    del arg155_1
    buf256 = buf232; del buf232  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___6___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf255, (1568, 512), (512, 1), 0), reinterpret_tensor(arg156_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf256)
    del arg156_1
    del arg157_1
    buf257 = reinterpret_tensor(buf255, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf255  # reuse
    buf258 = reinterpret_tensor(buf243, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf243  # reuse
    cpp_fused_clone_mul_64(c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    buf259 = reinterpret_tensor(buf239, (512, 49, 49), (2401, 49, 1), 0); del buf239  # reuse
    # Source Nodes: [attn_50], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf257, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf258, (512, 32, 49), (1568, 49, 1), 0), out=buf259)
    buf260 = buf238; del buf238  # reuse
    buf261 = reinterpret_tensor(buf259, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf259  # reuse
    buf262 = buf236; del buf236  # reuse
    buf263 = buf261; del buf261  # reuse
    buf264 = reinterpret_tensor(buf258, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf258  # reuse
    cpp_fused__softmax_add_clone_65(c_void_p(buf263.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()))
    del arg10_1
    del arg344_1
    buf265 = reinterpret_tensor(buf257, (512, 49, 32), (1568, 32, 1), 0); del buf257  # reuse
    # Source Nodes: [x_197], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf263, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf264, (512, 49, 32), (1568, 32, 1), 0), out=buf265)
    buf266 = reinterpret_tensor(buf264, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf264  # reuse
    cpp_fused_clone_66(c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf265, (1568, 512), (512, 1), 0); del buf265  # reuse
    # Source Nodes: [x_199], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf266, (1568, 512), (512, 1), 0), reinterpret_tensor(arg158_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf267)
    del arg158_1
    del arg159_1
    buf268 = reinterpret_tensor(buf253, (8, 196, 1), (196, 1, 1568), 0); del buf253  # reuse
    buf269 = reinterpret_tensor(buf252, (8, 196, 1), (196, 1, 1568), 0); del buf252  # reuse
    buf271 = reinterpret_tensor(buf266, (8, 196, 512), (100352, 512, 1), 0); del buf266  # reuse
    cpp_fused_native_layer_norm_67(c_void_p(buf251.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()))
    del arg160_1
    del arg161_1
    buf272 = reinterpret_tensor(buf249, (1568, 2048), (2048, 1), 0); del buf249  # reuse
    # Source Nodes: [x_206], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg163_1, reinterpret_tensor(buf271, (1568, 512), (512, 1), 0), reinterpret_tensor(arg162_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf272)
    del arg162_1
    del arg163_1
    buf273 = reinterpret_tensor(buf272, (8, 196, 2048), (401408, 2048, 1), 0); del buf272  # reuse
    cpp_fused_gelu_68(c_void_p(buf273.data_ptr()))
    buf274 = reinterpret_tensor(buf271, (1568, 512), (512, 1), 0); del buf271  # reuse
    # Source Nodes: [x_210], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf273, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg164_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf274)
    del arg164_1
    del arg165_1
    buf275 = reinterpret_tensor(buf269, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf269  # reuse
    buf276 = reinterpret_tensor(buf268, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf268  # reuse
    buf278 = reinterpret_tensor(buf227, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf227  # reuse
    cpp_fused_clone_native_layer_norm_69(c_void_p(buf251.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf278.data_ptr()))
    del arg166_1
    del arg167_1
    buf279 = buf256; del buf256  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___7___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg169_1, reinterpret_tensor(buf278, (1568, 512), (512, 1), 0), reinterpret_tensor(arg168_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf279)
    del arg168_1
    del arg169_1
    buf280 = reinterpret_tensor(buf278, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf278  # reuse
    buf281 = reinterpret_tensor(buf220, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf220  # reuse
    cpp_fused_clone_mul_70(c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    buf282 = reinterpret_tensor(buf263, (512, 49, 49), (2401, 49, 1), 0); del buf263  # reuse
    # Source Nodes: [attn_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf280, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf281, (512, 32, 49), (1568, 49, 1), 0), out=buf282)
    buf283 = buf262; del buf262  # reuse
    buf284 = reinterpret_tensor(buf282, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf282  # reuse
    buf285 = buf260; del buf260  # reuse
    buf286 = buf284; del buf284  # reuse
    buf287 = reinterpret_tensor(buf281, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf281  # reuse
    cpp_fused__softmax_clone_71(c_void_p(buf286.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()))
    del arg11_1
    del arg345_1
    del arg346_1
    buf288 = reinterpret_tensor(buf280, (512, 49, 32), (1568, 32, 1), 0); del buf280  # reuse
    # Source Nodes: [x_215], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf286, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf287, (512, 49, 32), (1568, 32, 1), 0), out=buf288)
    buf289 = reinterpret_tensor(buf287, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf287  # reuse
    cpp_fused_clone_72(c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf288, (1568, 512), (512, 1), 0); del buf288  # reuse
    # Source Nodes: [x_217], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf289, (1568, 512), (512, 1), 0), reinterpret_tensor(arg170_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf290)
    del arg170_1
    del arg171_1
    buf291 = reinterpret_tensor(buf276, (8, 196, 1), (196, 1, 1568), 0); del buf276  # reuse
    buf292 = reinterpret_tensor(buf275, (8, 196, 1), (196, 1, 1568), 0); del buf275  # reuse
    buf294 = reinterpret_tensor(buf289, (8, 196, 512), (100352, 512, 1), 0); del buf289  # reuse
    cpp_fused_native_layer_norm_73(c_void_p(buf251.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf294.data_ptr()))
    del arg172_1
    del arg173_1
    buf295 = reinterpret_tensor(buf273, (1568, 2048), (2048, 1), 0); del buf273  # reuse
    # Source Nodes: [x_224], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf294, (1568, 512), (512, 1), 0), reinterpret_tensor(arg174_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf295)
    del arg174_1
    del arg175_1
    buf296 = reinterpret_tensor(buf295, (8, 196, 2048), (401408, 2048, 1), 0); del buf295  # reuse
    cpp_fused_gelu_74(c_void_p(buf296.data_ptr()))
    buf297 = reinterpret_tensor(buf294, (1568, 512), (512, 1), 0); del buf294  # reuse
    # Source Nodes: [x_228], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf296, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf297)
    del arg176_1
    del arg177_1
    buf298 = reinterpret_tensor(buf297, (8, 196, 512), (100352, 512, 1), 0); del buf297  # reuse
    buf299 = reinterpret_tensor(buf292, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf292  # reuse
    buf300 = reinterpret_tensor(buf291, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf291  # reuse
    buf302 = reinterpret_tensor(buf204, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf204  # reuse
    cpp_fused_add_clone_native_layer_norm_75(c_void_p(buf298.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()))
    del arg178_1
    del arg179_1
    buf303 = buf279; del buf279  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___8___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf302, (1568, 512), (512, 1), 0), reinterpret_tensor(arg180_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf303)
    del arg180_1
    del arg181_1
    buf304 = reinterpret_tensor(buf302, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf302  # reuse
    buf305 = reinterpret_tensor(buf290, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf290  # reuse
    cpp_fused_clone_mul_76(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    buf306 = reinterpret_tensor(buf286, (512, 49, 49), (2401, 49, 1), 0); del buf286  # reuse
    # Source Nodes: [attn_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf304, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf305, (512, 32, 49), (1568, 49, 1), 0), out=buf306)
    buf307 = buf285; del buf285  # reuse
    buf308 = reinterpret_tensor(buf306, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf306  # reuse
    buf309 = buf283; del buf283  # reuse
    buf310 = buf308; del buf308  # reuse
    buf311 = reinterpret_tensor(buf305, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf305  # reuse
    cpp_fused__softmax_add_clone_77(c_void_p(buf310.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()))
    del arg12_1
    del arg347_1
    buf312 = reinterpret_tensor(buf304, (512, 49, 32), (1568, 32, 1), 0); del buf304  # reuse
    # Source Nodes: [x_233], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf310, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf311, (512, 49, 32), (1568, 32, 1), 0), out=buf312)
    buf313 = reinterpret_tensor(buf311, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf311  # reuse
    cpp_fused_clone_78(c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = reinterpret_tensor(buf312, (1568, 512), (512, 1), 0); del buf312  # reuse
    # Source Nodes: [x_235], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf313, (1568, 512), (512, 1), 0), reinterpret_tensor(arg182_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf314)
    del arg182_1
    del arg183_1
    buf315 = reinterpret_tensor(buf300, (8, 196, 1), (196, 1, 1568), 0); del buf300  # reuse
    buf316 = reinterpret_tensor(buf299, (8, 196, 1), (196, 1, 1568), 0); del buf299  # reuse
    buf318 = reinterpret_tensor(buf313, (8, 196, 512), (100352, 512, 1), 0); del buf313  # reuse
    cpp_fused_native_layer_norm_79(c_void_p(buf298.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()))
    del arg184_1
    del arg185_1
    buf319 = reinterpret_tensor(buf296, (1568, 2048), (2048, 1), 0); del buf296  # reuse
    # Source Nodes: [x_242], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf318, (1568, 512), (512, 1), 0), reinterpret_tensor(arg186_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf319)
    del arg186_1
    del arg187_1
    buf320 = reinterpret_tensor(buf319, (8, 196, 2048), (401408, 2048, 1), 0); del buf319  # reuse
    cpp_fused_gelu_80(c_void_p(buf320.data_ptr()))
    buf321 = reinterpret_tensor(buf318, (1568, 512), (512, 1), 0); del buf318  # reuse
    # Source Nodes: [x_246], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg189_1, reinterpret_tensor(buf320, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg188_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf321)
    del arg188_1
    del arg189_1
    buf322 = reinterpret_tensor(buf316, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf316  # reuse
    buf323 = reinterpret_tensor(buf315, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf315  # reuse
    buf325 = reinterpret_tensor(buf274, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf274  # reuse
    cpp_fused_clone_native_layer_norm_81(c_void_p(buf298.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()))
    del arg190_1
    del arg191_1
    buf326 = buf303; del buf303  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___9___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf325, (1568, 512), (512, 1), 0), reinterpret_tensor(arg192_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf326)
    del arg192_1
    del arg193_1
    buf327 = reinterpret_tensor(buf325, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf325  # reuse
    buf328 = reinterpret_tensor(buf267, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf267  # reuse
    cpp_fused_clone_mul_82(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    buf329 = reinterpret_tensor(buf310, (512, 49, 49), (2401, 49, 1), 0); del buf310  # reuse
    # Source Nodes: [attn_64], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf327, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf328, (512, 32, 49), (1568, 49, 1), 0), out=buf329)
    buf330 = buf309; del buf309  # reuse
    buf331 = reinterpret_tensor(buf329, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf329  # reuse
    buf332 = buf307; del buf307  # reuse
    buf333 = buf331; del buf331  # reuse
    buf334 = reinterpret_tensor(buf328, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf328  # reuse
    cpp_fused__softmax_clone_83(c_void_p(buf333.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()))
    del arg13_1
    del arg348_1
    del arg349_1
    buf335 = reinterpret_tensor(buf327, (512, 49, 32), (1568, 32, 1), 0); del buf327  # reuse
    # Source Nodes: [x_251], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf333, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf334, (512, 49, 32), (1568, 32, 1), 0), out=buf335)
    buf336 = reinterpret_tensor(buf334, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf334  # reuse
    cpp_fused_clone_84(c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = reinterpret_tensor(buf335, (1568, 512), (512, 1), 0); del buf335  # reuse
    # Source Nodes: [x_253], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg195_1, reinterpret_tensor(buf336, (1568, 512), (512, 1), 0), reinterpret_tensor(arg194_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf337)
    del arg194_1
    del arg195_1
    buf338 = reinterpret_tensor(buf323, (8, 196, 1), (196, 1, 1568), 0); del buf323  # reuse
    buf339 = reinterpret_tensor(buf322, (8, 196, 1), (196, 1, 1568), 0); del buf322  # reuse
    buf341 = reinterpret_tensor(buf336, (8, 196, 512), (100352, 512, 1), 0); del buf336  # reuse
    cpp_fused_native_layer_norm_85(c_void_p(buf298.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()))
    del arg196_1
    del arg197_1
    buf342 = reinterpret_tensor(buf320, (1568, 2048), (2048, 1), 0); del buf320  # reuse
    # Source Nodes: [x_260], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg199_1, reinterpret_tensor(buf341, (1568, 512), (512, 1), 0), reinterpret_tensor(arg198_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf342)
    del arg198_1
    del arg199_1
    buf343 = reinterpret_tensor(buf342, (8, 196, 2048), (401408, 2048, 1), 0); del buf342  # reuse
    cpp_fused_gelu_86(c_void_p(buf343.data_ptr()))
    buf344 = reinterpret_tensor(buf341, (1568, 512), (512, 1), 0); del buf341  # reuse
    # Source Nodes: [x_264], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg201_1, reinterpret_tensor(buf343, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg200_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf344)
    del arg200_1
    del arg201_1
    buf345 = reinterpret_tensor(buf344, (8, 196, 512), (100352, 512, 1), 0); del buf344  # reuse
    buf346 = reinterpret_tensor(buf339, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf339  # reuse
    buf347 = reinterpret_tensor(buf338, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf338  # reuse
    buf349 = reinterpret_tensor(buf251, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf251  # reuse
    cpp_fused_add_clone_native_layer_norm_87(c_void_p(buf345.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf349.data_ptr()))
    del arg202_1
    del arg203_1
    buf350 = buf326; del buf326  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___10___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg205_1, reinterpret_tensor(buf349, (1568, 512), (512, 1), 0), reinterpret_tensor(arg204_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf350)
    del arg204_1
    del arg205_1
    buf351 = reinterpret_tensor(buf349, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf349  # reuse
    buf352 = reinterpret_tensor(buf337, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf337  # reuse
    cpp_fused_clone_mul_88(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    buf353 = reinterpret_tensor(buf333, (512, 49, 49), (2401, 49, 1), 0); del buf333  # reuse
    # Source Nodes: [attn_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf351, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf352, (512, 32, 49), (1568, 49, 1), 0), out=buf353)
    buf354 = buf332; del buf332  # reuse
    buf355 = reinterpret_tensor(buf353, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf353  # reuse
    buf356 = buf330; del buf330  # reuse
    buf357 = buf355; del buf355  # reuse
    buf358 = reinterpret_tensor(buf352, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf352  # reuse
    cpp_fused__softmax_add_clone_89(c_void_p(buf357.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()))
    del arg14_1
    del arg350_1
    buf359 = reinterpret_tensor(buf351, (512, 49, 32), (1568, 32, 1), 0); del buf351  # reuse
    # Source Nodes: [x_269], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf357, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf358, (512, 49, 32), (1568, 32, 1), 0), out=buf359)
    buf360 = reinterpret_tensor(buf358, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf358  # reuse
    cpp_fused_clone_90(c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()))
    buf361 = reinterpret_tensor(buf359, (1568, 512), (512, 1), 0); del buf359  # reuse
    # Source Nodes: [x_271], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg207_1, reinterpret_tensor(buf360, (1568, 512), (512, 1), 0), reinterpret_tensor(arg206_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf361)
    del arg206_1
    del arg207_1
    buf362 = reinterpret_tensor(buf347, (8, 196, 1), (196, 1, 1568), 0); del buf347  # reuse
    buf363 = reinterpret_tensor(buf346, (8, 196, 1), (196, 1, 1568), 0); del buf346  # reuse
    buf365 = reinterpret_tensor(buf360, (8, 196, 512), (100352, 512, 1), 0); del buf360  # reuse
    cpp_fused_native_layer_norm_91(c_void_p(buf345.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf365.data_ptr()))
    del arg208_1
    del arg209_1
    buf366 = reinterpret_tensor(buf343, (1568, 2048), (2048, 1), 0); del buf343  # reuse
    # Source Nodes: [x_278], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg211_1, reinterpret_tensor(buf365, (1568, 512), (512, 1), 0), reinterpret_tensor(arg210_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf366)
    del arg210_1
    del arg211_1
    buf367 = reinterpret_tensor(buf366, (8, 196, 2048), (401408, 2048, 1), 0); del buf366  # reuse
    cpp_fused_gelu_92(c_void_p(buf367.data_ptr()))
    buf368 = reinterpret_tensor(buf365, (1568, 512), (512, 1), 0); del buf365  # reuse
    # Source Nodes: [x_282], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg213_1, reinterpret_tensor(buf367, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg212_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf368)
    del arg212_1
    del arg213_1
    buf369 = reinterpret_tensor(buf363, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf363  # reuse
    buf370 = reinterpret_tensor(buf362, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf362  # reuse
    buf372 = reinterpret_tensor(buf321, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf321  # reuse
    cpp_fused_clone_native_layer_norm_93(c_void_p(buf345.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()))
    del arg214_1
    del arg215_1
    buf373 = buf350; del buf350  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___11___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg217_1, reinterpret_tensor(buf372, (1568, 512), (512, 1), 0), reinterpret_tensor(arg216_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf373)
    del arg216_1
    del arg217_1
    buf374 = reinterpret_tensor(buf372, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf372  # reuse
    buf375 = reinterpret_tensor(buf314, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf314  # reuse
    cpp_fused_clone_mul_94(c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    buf376 = reinterpret_tensor(buf357, (512, 49, 49), (2401, 49, 1), 0); del buf357  # reuse
    # Source Nodes: [attn_74], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf374, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf375, (512, 32, 49), (1568, 49, 1), 0), out=buf376)
    buf377 = buf356; del buf356  # reuse
    buf378 = reinterpret_tensor(buf376, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf376  # reuse
    buf379 = buf354; del buf354  # reuse
    buf380 = buf378; del buf378  # reuse
    buf381 = reinterpret_tensor(buf375, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf375  # reuse
    cpp_fused__softmax_clone_95(c_void_p(buf380.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()))
    del arg15_1
    del arg351_1
    del arg352_1
    buf382 = reinterpret_tensor(buf374, (512, 49, 32), (1568, 32, 1), 0); del buf374  # reuse
    # Source Nodes: [x_287], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf380, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf381, (512, 49, 32), (1568, 32, 1), 0), out=buf382)
    buf383 = reinterpret_tensor(buf381, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf381  # reuse
    cpp_fused_clone_96(c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    buf384 = reinterpret_tensor(buf382, (1568, 512), (512, 1), 0); del buf382  # reuse
    # Source Nodes: [x_289], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg219_1, reinterpret_tensor(buf383, (1568, 512), (512, 1), 0), reinterpret_tensor(arg218_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf384)
    del arg218_1
    del arg219_1
    buf385 = reinterpret_tensor(buf370, (8, 196, 1), (196, 1, 1568), 0); del buf370  # reuse
    buf386 = reinterpret_tensor(buf369, (8, 196, 1), (196, 1, 1568), 0); del buf369  # reuse
    buf388 = reinterpret_tensor(buf383, (8, 196, 512), (100352, 512, 1), 0); del buf383  # reuse
    cpp_fused_native_layer_norm_97(c_void_p(buf345.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()))
    del arg220_1
    del arg221_1
    buf389 = reinterpret_tensor(buf367, (1568, 2048), (2048, 1), 0); del buf367  # reuse
    # Source Nodes: [x_296], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg223_1, reinterpret_tensor(buf388, (1568, 512), (512, 1), 0), reinterpret_tensor(arg222_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf389)
    del arg222_1
    del arg223_1
    buf390 = reinterpret_tensor(buf389, (8, 196, 2048), (401408, 2048, 1), 0); del buf389  # reuse
    cpp_fused_gelu_98(c_void_p(buf390.data_ptr()))
    buf391 = reinterpret_tensor(buf388, (1568, 512), (512, 1), 0); del buf388  # reuse
    # Source Nodes: [x_300], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg225_1, reinterpret_tensor(buf390, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg224_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf391)
    del arg224_1
    del arg225_1
    buf392 = reinterpret_tensor(buf391, (8, 196, 512), (100352, 512, 1), 0); del buf391  # reuse
    buf393 = reinterpret_tensor(buf386, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf386  # reuse
    buf394 = reinterpret_tensor(buf385, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf385  # reuse
    buf396 = reinterpret_tensor(buf298, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf298  # reuse
    cpp_fused_add_clone_native_layer_norm_99(c_void_p(buf392.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf396.data_ptr()))
    del arg226_1
    del arg227_1
    buf397 = buf373; del buf373  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___12___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg229_1, reinterpret_tensor(buf396, (1568, 512), (512, 1), 0), reinterpret_tensor(arg228_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf397)
    del arg228_1
    del arg229_1
    buf398 = reinterpret_tensor(buf396, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf396  # reuse
    buf399 = reinterpret_tensor(buf384, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf384  # reuse
    cpp_fused_clone_mul_100(c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    buf400 = reinterpret_tensor(buf380, (512, 49, 49), (2401, 49, 1), 0); del buf380  # reuse
    # Source Nodes: [attn_80], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf398, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf399, (512, 32, 49), (1568, 49, 1), 0), out=buf400)
    buf401 = buf379; del buf379  # reuse
    buf402 = reinterpret_tensor(buf400, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf400  # reuse
    buf403 = buf377; del buf377  # reuse
    buf404 = buf402; del buf402  # reuse
    buf405 = reinterpret_tensor(buf399, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf399  # reuse
    cpp_fused__softmax_add_clone_101(c_void_p(buf404.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf405.data_ptr()))
    del arg16_1
    del arg353_1
    buf406 = reinterpret_tensor(buf398, (512, 49, 32), (1568, 32, 1), 0); del buf398  # reuse
    # Source Nodes: [x_305], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf404, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf405, (512, 49, 32), (1568, 32, 1), 0), out=buf406)
    buf407 = reinterpret_tensor(buf405, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf405  # reuse
    cpp_fused_clone_102(c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = reinterpret_tensor(buf406, (1568, 512), (512, 1), 0); del buf406  # reuse
    # Source Nodes: [x_307], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg231_1, reinterpret_tensor(buf407, (1568, 512), (512, 1), 0), reinterpret_tensor(arg230_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf408)
    del arg230_1
    del arg231_1
    buf409 = reinterpret_tensor(buf394, (8, 196, 1), (196, 1, 1568), 0); del buf394  # reuse
    buf410 = reinterpret_tensor(buf393, (8, 196, 1), (196, 1, 1568), 0); del buf393  # reuse
    buf412 = reinterpret_tensor(buf407, (8, 196, 512), (100352, 512, 1), 0); del buf407  # reuse
    cpp_fused_native_layer_norm_103(c_void_p(buf392.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf412.data_ptr()))
    del arg232_1
    del arg233_1
    buf413 = reinterpret_tensor(buf390, (1568, 2048), (2048, 1), 0); del buf390  # reuse
    # Source Nodes: [x_314], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg235_1, reinterpret_tensor(buf412, (1568, 512), (512, 1), 0), reinterpret_tensor(arg234_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf413)
    del arg234_1
    del arg235_1
    buf414 = reinterpret_tensor(buf413, (8, 196, 2048), (401408, 2048, 1), 0); del buf413  # reuse
    cpp_fused_gelu_104(c_void_p(buf414.data_ptr()))
    buf415 = reinterpret_tensor(buf412, (1568, 512), (512, 1), 0); del buf412  # reuse
    # Source Nodes: [x_318], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg237_1, reinterpret_tensor(buf414, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg236_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf415)
    del arg236_1
    del arg237_1
    buf416 = reinterpret_tensor(buf410, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf410  # reuse
    buf417 = reinterpret_tensor(buf409, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf409  # reuse
    buf419 = reinterpret_tensor(buf368, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf368  # reuse
    cpp_fused_clone_native_layer_norm_105(c_void_p(buf392.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()))
    del arg238_1
    del arg239_1
    buf420 = buf397; del buf397  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___13___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg241_1, reinterpret_tensor(buf419, (1568, 512), (512, 1), 0), reinterpret_tensor(arg240_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf420)
    del arg240_1
    del arg241_1
    buf421 = reinterpret_tensor(buf419, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf419  # reuse
    buf422 = reinterpret_tensor(buf361, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf361  # reuse
    cpp_fused_clone_mul_106(c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()))
    buf423 = reinterpret_tensor(buf404, (512, 49, 49), (2401, 49, 1), 0); del buf404  # reuse
    # Source Nodes: [attn_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf421, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf422, (512, 32, 49), (1568, 49, 1), 0), out=buf423)
    buf424 = buf403; del buf403  # reuse
    buf425 = reinterpret_tensor(buf423, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf423  # reuse
    buf426 = buf401; del buf401  # reuse
    buf427 = buf425; del buf425  # reuse
    buf428 = reinterpret_tensor(buf422, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf422  # reuse
    cpp_fused__softmax_clone_107(c_void_p(buf427.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()))
    del arg17_1
    del arg354_1
    del arg355_1
    buf429 = reinterpret_tensor(buf421, (512, 49, 32), (1568, 32, 1), 0); del buf421  # reuse
    # Source Nodes: [x_323], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf427, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf428, (512, 49, 32), (1568, 32, 1), 0), out=buf429)
    buf430 = reinterpret_tensor(buf428, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf428  # reuse
    cpp_fused_clone_108(c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()))
    buf431 = reinterpret_tensor(buf429, (1568, 512), (512, 1), 0); del buf429  # reuse
    # Source Nodes: [x_325], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg243_1, reinterpret_tensor(buf430, (1568, 512), (512, 1), 0), reinterpret_tensor(arg242_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf431)
    del arg242_1
    del arg243_1
    buf432 = reinterpret_tensor(buf417, (8, 196, 1), (196, 1, 1568), 0); del buf417  # reuse
    buf433 = reinterpret_tensor(buf416, (8, 196, 1), (196, 1, 1568), 0); del buf416  # reuse
    buf435 = reinterpret_tensor(buf430, (8, 196, 512), (100352, 512, 1), 0); del buf430  # reuse
    cpp_fused_native_layer_norm_109(c_void_p(buf392.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf435.data_ptr()))
    del arg244_1
    del arg245_1
    buf436 = reinterpret_tensor(buf414, (1568, 2048), (2048, 1), 0); del buf414  # reuse
    # Source Nodes: [x_332], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg247_1, reinterpret_tensor(buf435, (1568, 512), (512, 1), 0), reinterpret_tensor(arg246_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf436)
    del arg246_1
    del arg247_1
    buf437 = reinterpret_tensor(buf436, (8, 196, 2048), (401408, 2048, 1), 0); del buf436  # reuse
    cpp_fused_gelu_110(c_void_p(buf437.data_ptr()))
    buf438 = reinterpret_tensor(buf435, (1568, 512), (512, 1), 0); del buf435  # reuse
    # Source Nodes: [x_336], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg249_1, reinterpret_tensor(buf437, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg248_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf438)
    del arg248_1
    del arg249_1
    buf439 = reinterpret_tensor(buf438, (8, 196, 512), (100352, 512, 1), 0); del buf438  # reuse
    buf440 = reinterpret_tensor(buf433, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf433  # reuse
    buf441 = reinterpret_tensor(buf432, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf432  # reuse
    buf443 = reinterpret_tensor(buf345, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf345  # reuse
    cpp_fused_add_clone_native_layer_norm_111(c_void_p(buf439.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf443.data_ptr()))
    del arg250_1
    del arg251_1
    buf444 = buf420; del buf420  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___14___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg253_1, reinterpret_tensor(buf443, (1568, 512), (512, 1), 0), reinterpret_tensor(arg252_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf444)
    del arg252_1
    del arg253_1
    buf445 = reinterpret_tensor(buf443, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf443  # reuse
    buf446 = reinterpret_tensor(buf431, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf431  # reuse
    cpp_fused_clone_mul_112(c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()))
    buf447 = reinterpret_tensor(buf427, (512, 49, 49), (2401, 49, 1), 0); del buf427  # reuse
    # Source Nodes: [attn_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf445, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf446, (512, 32, 49), (1568, 49, 1), 0), out=buf447)
    buf448 = buf426; del buf426  # reuse
    buf449 = reinterpret_tensor(buf447, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf447  # reuse
    buf450 = buf424; del buf424  # reuse
    buf451 = buf449; del buf449  # reuse
    buf452 = reinterpret_tensor(buf446, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf446  # reuse
    cpp_fused__softmax_add_clone_113(c_void_p(buf451.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()))
    del arg18_1
    del arg356_1
    buf453 = reinterpret_tensor(buf445, (512, 49, 32), (1568, 32, 1), 0); del buf445  # reuse
    # Source Nodes: [x_341], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf451, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf452, (512, 49, 32), (1568, 32, 1), 0), out=buf453)
    buf454 = reinterpret_tensor(buf452, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf452  # reuse
    cpp_fused_clone_114(c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()))
    buf455 = reinterpret_tensor(buf453, (1568, 512), (512, 1), 0); del buf453  # reuse
    # Source Nodes: [x_343], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg255_1, reinterpret_tensor(buf454, (1568, 512), (512, 1), 0), reinterpret_tensor(arg254_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf455)
    del arg254_1
    del arg255_1
    buf456 = reinterpret_tensor(buf441, (8, 196, 1), (196, 1, 1568), 0); del buf441  # reuse
    buf457 = reinterpret_tensor(buf440, (8, 196, 1), (196, 1, 1568), 0); del buf440  # reuse
    buf459 = reinterpret_tensor(buf454, (8, 196, 512), (100352, 512, 1), 0); del buf454  # reuse
    cpp_fused_native_layer_norm_115(c_void_p(buf439.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf459.data_ptr()))
    del arg256_1
    del arg257_1
    buf460 = reinterpret_tensor(buf437, (1568, 2048), (2048, 1), 0); del buf437  # reuse
    # Source Nodes: [x_350], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg259_1, reinterpret_tensor(buf459, (1568, 512), (512, 1), 0), reinterpret_tensor(arg258_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf460)
    del arg258_1
    del arg259_1
    buf461 = reinterpret_tensor(buf460, (8, 196, 2048), (401408, 2048, 1), 0); del buf460  # reuse
    cpp_fused_gelu_116(c_void_p(buf461.data_ptr()))
    buf462 = reinterpret_tensor(buf459, (1568, 512), (512, 1), 0); del buf459  # reuse
    # Source Nodes: [x_354], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg261_1, reinterpret_tensor(buf461, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg260_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf462)
    del arg260_1
    del arg261_1
    buf463 = reinterpret_tensor(buf457, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf457  # reuse
    buf464 = reinterpret_tensor(buf456, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf456  # reuse
    buf466 = reinterpret_tensor(buf415, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf415  # reuse
    cpp_fused_clone_native_layer_norm_117(c_void_p(buf439.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf466.data_ptr()))
    del arg262_1
    del arg263_1
    buf467 = buf444; del buf444  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___15___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg265_1, reinterpret_tensor(buf466, (1568, 512), (512, 1), 0), reinterpret_tensor(arg264_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf467)
    del arg264_1
    del arg265_1
    buf468 = reinterpret_tensor(buf466, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf466  # reuse
    buf469 = reinterpret_tensor(buf408, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf408  # reuse
    cpp_fused_clone_mul_118(c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()))
    buf470 = reinterpret_tensor(buf451, (512, 49, 49), (2401, 49, 1), 0); del buf451  # reuse
    # Source Nodes: [attn_94], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf468, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf469, (512, 32, 49), (1568, 49, 1), 0), out=buf470)
    buf471 = buf450; del buf450  # reuse
    buf472 = reinterpret_tensor(buf470, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf470  # reuse
    buf473 = buf448; del buf448  # reuse
    buf474 = buf472; del buf472  # reuse
    buf475 = reinterpret_tensor(buf469, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf469  # reuse
    cpp_fused__softmax_clone_119(c_void_p(buf474.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf475.data_ptr()))
    del arg19_1
    del arg357_1
    del arg358_1
    buf476 = reinterpret_tensor(buf468, (512, 49, 32), (1568, 32, 1), 0); del buf468  # reuse
    # Source Nodes: [x_359], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf474, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf475, (512, 49, 32), (1568, 32, 1), 0), out=buf476)
    buf477 = reinterpret_tensor(buf475, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf475  # reuse
    cpp_fused_clone_120(c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    buf478 = reinterpret_tensor(buf476, (1568, 512), (512, 1), 0); del buf476  # reuse
    # Source Nodes: [x_361], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg267_1, reinterpret_tensor(buf477, (1568, 512), (512, 1), 0), reinterpret_tensor(arg266_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf478)
    del arg266_1
    del arg267_1
    buf479 = reinterpret_tensor(buf464, (8, 196, 1), (196, 1, 1568), 0); del buf464  # reuse
    buf480 = reinterpret_tensor(buf463, (8, 196, 1), (196, 1, 1568), 0); del buf463  # reuse
    buf482 = reinterpret_tensor(buf477, (8, 196, 512), (100352, 512, 1), 0); del buf477  # reuse
    cpp_fused_native_layer_norm_121(c_void_p(buf439.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf482.data_ptr()))
    del arg268_1
    del arg269_1
    buf483 = reinterpret_tensor(buf461, (1568, 2048), (2048, 1), 0); del buf461  # reuse
    # Source Nodes: [x_368], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg271_1, reinterpret_tensor(buf482, (1568, 512), (512, 1), 0), reinterpret_tensor(arg270_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf483)
    del arg270_1
    del arg271_1
    buf484 = reinterpret_tensor(buf483, (8, 196, 2048), (401408, 2048, 1), 0); del buf483  # reuse
    cpp_fused_gelu_122(c_void_p(buf484.data_ptr()))
    buf485 = reinterpret_tensor(buf482, (1568, 512), (512, 1), 0); del buf482  # reuse
    # Source Nodes: [x_372], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg273_1, reinterpret_tensor(buf484, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg272_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf485)
    del arg272_1
    del arg273_1
    buf486 = reinterpret_tensor(buf485, (8, 196, 512), (100352, 512, 1), 0); del buf485  # reuse
    buf487 = reinterpret_tensor(buf480, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf480  # reuse
    buf488 = reinterpret_tensor(buf479, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf479  # reuse
    buf490 = reinterpret_tensor(buf392, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf392  # reuse
    cpp_fused_add_clone_native_layer_norm_123(c_void_p(buf486.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf490.data_ptr()))
    del arg274_1
    del arg275_1
    buf491 = buf467; del buf467  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___16___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg277_1, reinterpret_tensor(buf490, (1568, 512), (512, 1), 0), reinterpret_tensor(arg276_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf491)
    del arg276_1
    del arg277_1
    buf492 = reinterpret_tensor(buf490, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf490  # reuse
    buf493 = reinterpret_tensor(buf478, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf478  # reuse
    cpp_fused_clone_mul_124(c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()))
    buf494 = reinterpret_tensor(buf474, (512, 49, 49), (2401, 49, 1), 0); del buf474  # reuse
    # Source Nodes: [attn_100], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf492, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf493, (512, 32, 49), (1568, 49, 1), 0), out=buf494)
    buf495 = buf473; del buf473  # reuse
    buf496 = reinterpret_tensor(buf494, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf494  # reuse
    buf497 = buf471; del buf471  # reuse
    buf498 = buf496; del buf496  # reuse
    buf499 = reinterpret_tensor(buf493, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf493  # reuse
    cpp_fused__softmax_add_clone_125(c_void_p(buf498.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()))
    del arg20_1
    del arg359_1
    buf500 = reinterpret_tensor(buf492, (512, 49, 32), (1568, 32, 1), 0); del buf492  # reuse
    # Source Nodes: [x_377], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf498, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf499, (512, 49, 32), (1568, 32, 1), 0), out=buf500)
    buf501 = reinterpret_tensor(buf499, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf499  # reuse
    cpp_fused_clone_126(c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    buf502 = reinterpret_tensor(buf500, (1568, 512), (512, 1), 0); del buf500  # reuse
    # Source Nodes: [x_379], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg279_1, reinterpret_tensor(buf501, (1568, 512), (512, 1), 0), reinterpret_tensor(arg278_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf502)
    del arg278_1
    del arg279_1
    buf503 = reinterpret_tensor(buf488, (8, 196, 1), (196, 1, 1568), 0); del buf488  # reuse
    buf504 = reinterpret_tensor(buf487, (8, 196, 1), (196, 1, 1568), 0); del buf487  # reuse
    buf506 = reinterpret_tensor(buf501, (8, 196, 512), (100352, 512, 1), 0); del buf501  # reuse
    cpp_fused_native_layer_norm_127(c_void_p(buf486.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf506.data_ptr()))
    del arg280_1
    del arg281_1
    buf507 = reinterpret_tensor(buf484, (1568, 2048), (2048, 1), 0); del buf484  # reuse
    # Source Nodes: [x_386], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg283_1, reinterpret_tensor(buf506, (1568, 512), (512, 1), 0), reinterpret_tensor(arg282_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf507)
    del arg282_1
    del arg283_1
    buf508 = reinterpret_tensor(buf507, (8, 196, 2048), (401408, 2048, 1), 0); del buf507  # reuse
    cpp_fused_gelu_128(c_void_p(buf508.data_ptr()))
    buf509 = reinterpret_tensor(buf506, (1568, 512), (512, 1), 0); del buf506  # reuse
    # Source Nodes: [x_390], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg285_1, reinterpret_tensor(buf508, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf509)
    del arg284_1
    del arg285_1
    buf510 = reinterpret_tensor(buf504, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf504  # reuse
    buf511 = reinterpret_tensor(buf503, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf503  # reuse
    buf513 = reinterpret_tensor(buf462, (8, 2, 2, 7, 7, 512), (100352, 50176, 25088, 3584, 512, 1), 0); del buf462  # reuse
    cpp_fused_clone_native_layer_norm_129(c_void_p(buf486.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf513.data_ptr()))
    del arg286_1
    del arg287_1
    buf514 = buf491; del buf491  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___2___blocks___17___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg289_1, reinterpret_tensor(buf513, (1568, 512), (512, 1), 0), reinterpret_tensor(arg288_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf514)
    del arg288_1
    del arg289_1
    buf515 = reinterpret_tensor(buf513, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf513  # reuse
    buf516 = reinterpret_tensor(buf455, (32, 16, 32, 49), (25088, 1568, 49, 1), 0); del buf455  # reuse
    cpp_fused_clone_mul_130(c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    buf517 = reinterpret_tensor(buf498, (512, 49, 49), (2401, 49, 1), 0); del buf498  # reuse
    # Source Nodes: [attn_104], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf515, (512, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf516, (512, 32, 49), (1568, 49, 1), 0), out=buf517)
    buf518 = buf497; del buf497  # reuse
    buf519 = reinterpret_tensor(buf517, (32, 16, 49, 49), (38416, 2401, 49, 1), 0); del buf517  # reuse
    buf520 = buf495; del buf495  # reuse
    buf521 = buf519; del buf519  # reuse
    buf522 = reinterpret_tensor(buf516, (32, 16, 49, 32), (25088, 1568, 32, 1), 0); del buf516  # reuse
    cpp_fused__softmax_clone_131(c_void_p(buf521.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf522.data_ptr()))
    del arg21_1
    del arg360_1
    del arg361_1
    del buf514
    del buf518
    del buf520
    buf523 = reinterpret_tensor(buf515, (512, 49, 32), (1568, 32, 1), 0); del buf515  # reuse
    # Source Nodes: [x_395], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf521, (512, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf522, (512, 49, 32), (1568, 32, 1), 0), out=buf523)
    del buf521
    buf524 = reinterpret_tensor(buf522, (32, 49, 16, 32), (25088, 512, 32, 1), 0); del buf522  # reuse
    cpp_fused_clone_132(c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()))
    buf525 = reinterpret_tensor(buf523, (1568, 512), (512, 1), 0); del buf523  # reuse
    # Source Nodes: [x_397], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg291_1, reinterpret_tensor(buf524, (1568, 512), (512, 1), 0), reinterpret_tensor(arg290_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf525)
    del arg290_1
    del arg291_1
    buf526 = reinterpret_tensor(buf511, (8, 196, 1), (196, 1, 1568), 0); del buf511  # reuse
    buf527 = reinterpret_tensor(buf510, (8, 196, 1), (196, 1, 1568), 0); del buf510  # reuse
    buf529 = reinterpret_tensor(buf524, (8, 196, 512), (100352, 512, 1), 0); del buf524  # reuse
    cpp_fused_native_layer_norm_133(c_void_p(buf486.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf529.data_ptr()))
    del arg292_1
    del arg293_1
    del buf526
    del buf527
    buf530 = reinterpret_tensor(buf508, (1568, 2048), (2048, 1), 0); del buf508  # reuse
    # Source Nodes: [x_404], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg295_1, reinterpret_tensor(buf529, (1568, 512), (512, 1), 0), reinterpret_tensor(arg294_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf530)
    del arg294_1
    del arg295_1
    buf531 = reinterpret_tensor(buf530, (8, 196, 2048), (401408, 2048, 1), 0); del buf530  # reuse
    cpp_fused_gelu_134(c_void_p(buf531.data_ptr()))
    buf532 = reinterpret_tensor(buf529, (1568, 512), (512, 1), 0); del buf529  # reuse
    # Source Nodes: [x_408], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg297_1, reinterpret_tensor(buf531, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg296_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf532)
    del arg296_1
    del arg297_1
    del buf531
    buf533 = reinterpret_tensor(buf532, (8, 7, 7, 2, 2, 512), (100352, 14336, 1024, 512, 7168, 1), 0); del buf532  # reuse
    buf534 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cpu', dtype=torch.float32)
    buf535 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cpu', dtype=torch.float32)
    buf537 = reinterpret_tensor(buf439, (8, 7, 7, 2048), (100352, 14336, 2048, 1), 0); del buf439  # reuse
    cpp_fused_clone_native_layer_norm_135(c_void_p(buf533.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf537.data_ptr()))
    del arg298_1
    del arg299_1
    del buf486
    del buf502
    del buf509
    del buf525
    del buf533
    buf538 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_416], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf537, (392, 2048), (2048, 1), 0), reinterpret_tensor(arg300_1, (2048, 1024), (1, 2048), 0), out=buf538)
    del arg300_1
    del buf537
    buf539 = buf535; del buf535  # reuse
    buf540 = buf534; del buf534  # reuse
    buf542 = empty((8, 7, 7, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_136(c_void_p(buf538.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf542.data_ptr()))
    del arg301_1
    del arg302_1
    buf543 = reinterpret_tensor(buf0, (392, 3072), (3072, 1), 0); del buf0  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg304_1, reinterpret_tensor(buf542, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg303_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf543)
    del arg303_1
    del arg304_1
    buf544 = reinterpret_tensor(buf542, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf542  # reuse
    buf545 = empty((8, 32, 32, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_137(c_void_p(buf543.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()))
    buf546 = empty((256, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_110], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf544, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf545, (256, 32, 49), (1568, 49, 1), 0), out=buf546)
    buf547 = empty_strided((8, 32, 49, 1), (1568, 49, 1, 12544), device='cpu', dtype=torch.float32)
    buf548 = reinterpret_tensor(buf546, (8, 32, 49, 49), (76832, 2401, 49, 1), 0); del buf546  # reuse
    buf549 = empty_strided((8, 32, 49, 1), (1568, 49, 1, 12544), device='cpu', dtype=torch.float32)
    buf550 = buf548; del buf548  # reuse
    buf551 = reinterpret_tensor(buf545, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf545  # reuse
    cpp_fused__softmax_add_clone_138(c_void_p(buf550.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf551.data_ptr()))
    del arg22_1
    del arg362_1
    buf552 = reinterpret_tensor(buf544, (256, 49, 32), (1568, 32, 1), 0); del buf544  # reuse
    # Source Nodes: [x_418], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf550, (256, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf551, (256, 49, 32), (1568, 32, 1), 0), out=buf552)
    buf553 = reinterpret_tensor(buf551, (8, 49, 32, 32), (50176, 1024, 32, 1), 0); del buf551  # reuse
    cpp_fused_clone_139(c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()))
    buf554 = reinterpret_tensor(buf552, (392, 1024), (1024, 1), 0); del buf552  # reuse
    # Source Nodes: [x_420], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg306_1, reinterpret_tensor(buf553, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg305_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf554)
    del arg305_1
    del arg306_1
    buf555 = reinterpret_tensor(buf540, (8, 49, 1), (49, 1, 392), 0); del buf540  # reuse
    buf556 = reinterpret_tensor(buf539, (8, 49, 1), (49, 1, 392), 0); del buf539  # reuse
    buf558 = reinterpret_tensor(buf553, (8, 49, 1024), (50176, 1024, 1), 0); del buf553  # reuse
    cpp_fused_native_layer_norm_140(c_void_p(buf538.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf558.data_ptr()))
    del arg307_1
    del arg308_1
    buf559 = reinterpret_tensor(buf109, (392, 4096), (4096, 1), 0); del buf109  # reuse
    # Source Nodes: [x_427], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg310_1, reinterpret_tensor(buf558, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg309_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf559)
    del arg309_1
    del arg310_1
    buf560 = reinterpret_tensor(buf559, (8, 49, 4096), (200704, 4096, 1), 0); del buf559  # reuse
    cpp_fused_gelu_141(c_void_p(buf560.data_ptr()))
    buf561 = reinterpret_tensor(buf558, (392, 1024), (1024, 1), 0); del buf558  # reuse
    # Source Nodes: [x_431], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg312_1, reinterpret_tensor(buf560, (392, 4096), (4096, 1), 0), reinterpret_tensor(arg311_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf561)
    del arg311_1
    del arg312_1
    buf562 = reinterpret_tensor(buf556, (8, 7, 7, 1), (49, 7, 1, 392), 0); del buf556  # reuse
    buf563 = reinterpret_tensor(buf555, (8, 7, 7, 1), (49, 7, 1, 392), 0); del buf555  # reuse
    buf565 = empty((8, 7, 7, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_142(c_void_p(buf538.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf565.data_ptr()))
    del arg313_1
    del arg314_1
    buf566 = buf543; del buf543  # reuse
    # Source Nodes: [getattr_getattr_l__mod___layers___3___blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg316_1, reinterpret_tensor(buf565, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg315_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf566)
    del arg315_1
    del arg316_1
    buf567 = reinterpret_tensor(buf565, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf565  # reuse
    buf568 = empty((8, 32, 32, 49), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_143(c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()))
    buf569 = reinterpret_tensor(buf550, (256, 49, 49), (2401, 49, 1), 0); del buf550  # reuse
    # Source Nodes: [attn_114], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf567, (256, 49, 32), (1568, 32, 1), 0), reinterpret_tensor(buf568, (256, 32, 49), (1568, 49, 1), 0), out=buf569)
    buf570 = buf549; del buf549  # reuse
    buf571 = reinterpret_tensor(buf569, (8, 32, 49, 49), (76832, 2401, 49, 1), 0); del buf569  # reuse
    buf572 = buf547; del buf547  # reuse
    buf573 = buf571; del buf571  # reuse
    buf574 = reinterpret_tensor(buf568, (8, 32, 49, 32), (50176, 1568, 32, 1), 0); del buf568  # reuse
    cpp_fused__softmax_add_clone_144(c_void_p(buf573.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf574.data_ptr()))
    del arg23_1
    del arg363_1
    del buf566
    del buf570
    del buf572
    buf575 = reinterpret_tensor(buf567, (256, 49, 32), (1568, 32, 1), 0); del buf567  # reuse
    # Source Nodes: [x_436], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf573, (256, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf574, (256, 49, 32), (1568, 32, 1), 0), out=buf575)
    del buf573
    buf576 = reinterpret_tensor(buf574, (8, 49, 32, 32), (50176, 1024, 32, 1), 0); del buf574  # reuse
    cpp_fused_clone_145(c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()))
    buf577 = reinterpret_tensor(buf575, (392, 1024), (1024, 1), 0); del buf575  # reuse
    # Source Nodes: [x_438], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg318_1, reinterpret_tensor(buf576, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg317_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf577)
    del arg317_1
    del arg318_1
    buf578 = reinterpret_tensor(buf563, (8, 49, 1), (49, 1, 392), 0); del buf563  # reuse
    buf579 = reinterpret_tensor(buf562, (8, 49, 1), (49, 1, 392), 0); del buf562  # reuse
    buf581 = reinterpret_tensor(buf576, (8, 49, 1024), (50176, 1024, 1), 0); del buf576  # reuse
    cpp_fused_native_layer_norm_146(c_void_p(buf538.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf581.data_ptr()))
    del arg319_1
    del arg320_1
    buf582 = reinterpret_tensor(buf560, (392, 4096), (4096, 1), 0); del buf560  # reuse
    # Source Nodes: [x_445], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg322_1, reinterpret_tensor(buf581, (392, 1024), (1024, 1), 0), reinterpret_tensor(arg321_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf582)
    del arg321_1
    del arg322_1
    buf583 = reinterpret_tensor(buf582, (8, 49, 4096), (200704, 4096, 1), 0); del buf582  # reuse
    cpp_fused_gelu_147(c_void_p(buf583.data_ptr()))
    buf584 = reinterpret_tensor(buf581, (392, 1024), (1024, 1), 0); del buf581  # reuse
    # Source Nodes: [x_449], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg324_1, reinterpret_tensor(buf583, (392, 4096), (4096, 1), 0), reinterpret_tensor(arg323_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf584)
    del arg323_1
    del arg324_1
    del buf583
    buf585 = reinterpret_tensor(buf584, (8, 49, 1024), (50176, 1024, 1), 0); del buf584  # reuse
    buf586 = reinterpret_tensor(buf579, (8, 7, 7, 1), (49, 7, 1, 392), 0); del buf579  # reuse
    buf587 = reinterpret_tensor(buf578, (8, 7, 7, 1), (49, 7, 1, 392), 0); del buf578  # reuse
    buf589 = empty((8, 1024), device='cpu', dtype=torch.float32)
    buf590 = buf589; del buf589  # reuse
    cpp_fused_add_mean_native_layer_norm_148(c_void_p(buf585.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()))
    del arg325_1
    del arg326_1
    del buf538
    del buf554
    del buf561
    del buf577
    del buf585
    del buf586
    del buf587
    buf591 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_456, x_457, x_461], Original ATen: [aten.addmm, aten.mean, aten.native_layer_norm]
    extern_kernels.addmm(arg328_1, buf590, reinterpret_tensor(arg327_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf591)
    del arg327_1
    del arg328_1
    return (buf591, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((169, 4), (4, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((169, 4), (4, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((169, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((169, 8), (8, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((169, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((169, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((169, 32), (32, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((1024, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg330_1 = rand_strided((64, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg332_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg333_1 = rand_strided((16, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg335_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg336_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg338_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg339_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg341_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg342_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg344_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg345_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg347_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg348_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg350_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg351_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg353_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg354_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg356_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg357_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg359_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg360_1 = rand_strided((4, 49, 49), (2401, 49, 1), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg362_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg363_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg364_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swin_base_patch4_window7_224', benchmark_compiled_module)
