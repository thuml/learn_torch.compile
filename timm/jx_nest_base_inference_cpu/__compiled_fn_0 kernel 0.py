
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


cpp_fused_add_native_layer_norm_1 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(4L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 4L))) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*x2) + (25088L*x1)));
                                auto tmp2 = tmp0 + tmp1;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(4L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 4L))) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*x2) + (25088L*x1)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(128.0);
                            auto tmp8 = tmp6 / tmp7;
                            auto tmp9 = static_cast<float>(1e-06);
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = 1 / std::sqrt(tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp17 = tmp15 + tmp16;
                            tmp17.store(out_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (384L*x2) + (1204224L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (100352L*x1) + (401408L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(192L); x4+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(128L + x3 + (32L*x1) + (384L*x4) + (75264L*x2) + (1204224L*x0)), static_cast<long>(384L), tmp0, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x3_inner));
                                    auto tmp2 = static_cast<float>(0.42044820762685725);
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 * tmp3;
                                    tmp4.store(out_ptr1 + static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (100352L*x1) + (401408L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(192L); x4<static_cast<long>(196L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (32L*x1) + (384L*x4) + (75264L*x2) + (1204224L*x0)));
                                auto tmp1 = static_cast<float>(0.42044820762685725);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr1[static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (100352L*x1) + (401408L*x0))] = tmpbuf[x3_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x3 + (32L*x1) + (384L*x2) + (1204224L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (100352L*x1) + (401408L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x2) + (401408L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (401408L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_5 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(4L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 4L))) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*x2) + (25088L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(4L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 4L))) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*x2) + (25088L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 - tmp6;
                            auto tmp9 = static_cast<float>(128.0);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = static_cast<float>(1e-06);
                            auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                            auto tmp13 = 1 / std::sqrt(tmp12);
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp7 * tmp14;
                            auto tmp17 = tmp15 * tmp16;
                            auto tmp19 = tmp17 + tmp18;
                            tmp19.store(out_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                        }
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


cpp_fused_add_native_layer_norm_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(4L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 4L))) + (401408L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*x2) + (25088L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = static_cast<float>(tmp_acc0.m2);
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(4L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 4L))) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*x2) + (25088L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 - tmp8;
                            auto tmp11 = static_cast<float>(128.0);
                            auto tmp12 = tmp10 / tmp11;
                            auto tmp13 = static_cast<float>(1e-06);
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            auto tmp15 = 1 / std::sqrt(tmp14);
                            auto tmp16 = at::vec::Vectorized<float>(tmp15);
                            auto tmp17 = tmp9 * tmp16;
                            auto tmp19 = tmp17 * tmp18;
                            auto tmp21 = tmp19 + tmp20;
                            tmp21.store(out_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (384L*x2) + (1204224L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (100352L*x1) + (401408L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(192L); x4+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(128L + x3 + (32L*x1) + (384L*x4) + (75264L*x2) + (1204224L*x0)), static_cast<long>(384L), tmp0, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x3_inner));
                                    auto tmp2 = static_cast<float>(0.42044820762685725);
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 * tmp3;
                                    tmp4.store(out_ptr1 + static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (100352L*x1) + (401408L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(192L); x4<static_cast<long>(196L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (32L*x1) + (384L*x4) + (75264L*x2) + (1204224L*x0)));
                                auto tmp1 = static_cast<float>(0.42044820762685725);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr1[static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (100352L*x1) + (401408L*x0))] = tmpbuf[x3_inner]; }
                            }
                        }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x3 + (32L*x1) + (384L*x2) + (1204224L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (100352L*x1) + (401408L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x2) + (401408L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (401408L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_11 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(4L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 4L))) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*x2) + (25088L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            tmp8.store(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
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
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(128.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_clone_convolution_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(1792L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1792L*x2) + (25088L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (1792L*x2) + (25088L*x1) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (1792L*x1) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_14 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(57L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(57L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(56);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0)), to_float_mask(tmp5));
                                auto tmp8 = out_ptr0[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                                auto tmp12 = static_cast<float>(256.0);
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = static_cast<float>(1e-06);
                                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                                auto tmp16 = 1 / std::sqrt(tmp15);
                                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                auto tmp18 = tmp10 * tmp17;
                                auto tmp19 = masked_load(in_ptr1 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp20 = tmp18 * tmp19;
                                auto tmp21 = masked_load(in_ptr2 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp22 = tmp20 + tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp6(), to_float_mask(tmp5));
                            tmp23.store(out_ptr2 + static_cast<long>(x3 + (256L*x2) + (14592L*x1) + (831744L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(24L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(256L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(512L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(14592L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp8, 8);
                            float tmp11[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(14848L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp11, 8);
                            float tmp14[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(15104L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp14, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(29184L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp17, 8);
                            float tmp20[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(29440L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp20, 8);
                            float tmp23[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(29696L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)), static_cast<long>(512L), tmp23, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(tmp11 + static_cast<long>(8L*x1_inner));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(tmp14 + static_cast<long>(8L*x1_inner));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x1_inner));
                                auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                                auto tmp24 = at::vec::Vectorized<float>::loadu(tmp23 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = at::vec::maximum(tmp3, tmp1);
                                auto tmp7 = at::vec::maximum(tmp6, tmp4);
                                auto tmp10 = at::vec::maximum(tmp9, tmp7);
                                auto tmp13 = at::vec::maximum(tmp12, tmp10);
                                auto tmp16 = at::vec::maximum(tmp15, tmp13);
                                auto tmp19 = at::vec::maximum(tmp18, tmp16);
                                auto tmp22 = at::vec::maximum(tmp21, tmp19);
                                auto tmp25 = at::vec::maximum(tmp24, tmp22);
                                tmp25.store(out_ptr3 + static_cast<long>(x3 + (28L*x2) + (784L*x1) + (784L*x1_inner) + (200704L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(24L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(256L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(512L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(14592L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(14848L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(15104L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(29184L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(29440L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(29696L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            auto tmp8 = at::vec::maximum(tmp7, tmp6);
                            auto tmp10 = at::vec::maximum(tmp9, tmp8);
                            auto tmp12 = at::vec::maximum(tmp11, tmp10);
                            auto tmp14 = at::vec::maximum(tmp13, tmp12);
                            auto tmp16 = at::vec::maximum(tmp15, tmp14);
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x3 + (28L*x2) + (784L*x1) + (784L*x1_inner) + (200704L*x0))] = tmpbuf[x1_inner]; }
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                float tmp1[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)), static_cast<long>(256L), tmp1, 8);
                                at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)), static_cast<long>(256L), tmp1, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr3[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer((x2 + x2_inner), 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (784L*x3_inner) + (200704L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x3_inner));
                                    auto tmp3 = tmp0 + tmp2;
                                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                                }
                            }
                            tmp_acc0_vec.mean.store(out_ptr4 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                            tmp_acc0_vec.m2.store(out_ptr5 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = out_ptr3[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer(x2, 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (200704L*x0) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                                auto tmp1 = in_ptr3[static_cast<long>(x3 + (256L*x2) + (50176L*x1))];
                                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                                tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                            }
                            out_ptr4[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0.mean;
                            out_ptr5[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0.m2;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = out_ptr3[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer((x2 + x2_inner), 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (784L*x3_inner) + (200704L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1)));
                                auto tmp3 = out_ptr4[static_cast<long>(x2 + x2_inner + (196L*x1) + (784L*x0))];
                                auto tmp6 = out_ptr5[static_cast<long>(x2 + x2_inner + (196L*x1) + (784L*x0))];
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                                auto tmp5 = tmp2 - tmp4;
                                auto tmp7 = static_cast<float>(256.0);
                                auto tmp8 = tmp6 / tmp7;
                                auto tmp9 = static_cast<float>(1e-06);
                                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                                auto tmp11 = 1 / std::sqrt(tmp10);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp5 * tmp12;
                                auto tmp15 = tmp13 * tmp14;
                                auto tmp17 = tmp15 + tmp16;
                                tmp17.store(out_ptr6 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1) + (200704L*x0)));
                            }
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer(x2, 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (200704L*x0) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                            auto tmp1 = in_ptr3[static_cast<long>(x3 + (256L*x2) + (50176L*x1))];
                            auto tmp3 = out_ptr4[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp5 = out_ptr5[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp12 = in_ptr4[static_cast<long>(x3)];
                            auto tmp14 = in_ptr5[static_cast<long>(x3)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                            auto tmp6 = static_cast<float>(256.0);
                            auto tmp7 = tmp5 / tmp6;
                            auto tmp8 = static_cast<float>(1e-06);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                            auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                            auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                            out_ptr6[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))] = tmp15;
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (768L*x2) + (602112L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (25088L*x1) + (200704L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(192L); x4+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(256L + x3 + (32L*x1) + (768L*x4) + (150528L*x2) + (602112L*x0)), static_cast<long>(768L), tmp0, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x3_inner));
                                    auto tmp2 = static_cast<float>(0.42044820762685725);
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 * tmp3;
                                    tmp4.store(out_ptr1 + static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (25088L*x1) + (200704L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(192L); x4<static_cast<long>(196L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x3 + (32L*x1) + (768L*x4) + (150528L*x2) + (602112L*x0)));
                                auto tmp1 = static_cast<float>(0.42044820762685725);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr1[static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (25088L*x1) + (200704L*x0))] = tmpbuf[x3_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x3 + (32L*x1) + (768L*x2) + (602112L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (25088L*x1) + (200704L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (25088L*x2) + (200704L*x0)), static_cast<long>(25088L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (8L*x1) + (8L*x1_inner) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_18 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                float tmp1[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)), static_cast<long>(256L), tmp1, 8);
                                float tmp4[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)), static_cast<long>(256L), tmp4, 8);
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)), static_cast<long>(256L), tmp1, 8);
                                at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)), static_cast<long>(256L), tmp4, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer((x2 + x2_inner), 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (784L*x3_inner) + (200704L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x3_inner));
                                    auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x3_inner));
                                    auto tmp3 = tmp0 + tmp2;
                                    auto tmp6 = tmp3 + tmp5;
                                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                                }
                            }
                            tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                            tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer(x2, 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (200704L*x0) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (256L*x2) + (50176L*x1))];
                                auto tmp3 = in_ptr2[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                                tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                            }
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0.mean;
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0.m2;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer((x2 + x2_inner), 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (784L*x3_inner) + (200704L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1) + (200704L*x0)));
                                auto tmp5 = out_ptr0[static_cast<long>(x2 + x2_inner + (196L*x1) + (784L*x0))];
                                auto tmp8 = out_ptr1[static_cast<long>(x2 + x2_inner + (196L*x1) + (784L*x0))];
                                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                                auto tmp7 = tmp4 - tmp6;
                                auto tmp9 = static_cast<float>(256.0);
                                auto tmp10 = tmp8 / tmp9;
                                auto tmp11 = static_cast<float>(1e-06);
                                auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                                auto tmp13 = 1 / std::sqrt(tmp12);
                                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                auto tmp15 = tmp7 * tmp14;
                                auto tmp17 = tmp15 * tmp16;
                                auto tmp19 = tmp17 + tmp18;
                                tmp19.store(out_ptr2 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1) + (200704L*x0)));
                            }
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer(x2, 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (200704L*x0) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (256L*x2) + (50176L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))];
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp7 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp14 = in_ptr3[static_cast<long>(x3)];
                            auto tmp16 = in_ptr4[static_cast<long>(x3)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                            auto tmp8 = static_cast<float>(256.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            out_ptr2[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))] = tmp17;
                        }
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


cpp_fused_add_native_layer_norm_20 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                float tmp1[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)), static_cast<long>(256L), tmp1, 8);
                                float tmp4[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)), static_cast<long>(256L), tmp4, 8);
                                float tmp7[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)), static_cast<long>(256L), tmp7, 8);
                                at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)), static_cast<long>(256L), tmp1, 8);
                                at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)), static_cast<long>(256L), tmp4, 8);
                                at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)), static_cast<long>(256L), tmp7, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer((x2 + x2_inner), 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (784L*x3_inner) + (200704L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                    auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x3_inner));
                                    auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x3_inner));
                                    auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x3_inner));
                                    auto tmp3 = tmp0 + tmp2;
                                    auto tmp6 = tmp3 + tmp5;
                                    auto tmp9 = tmp6 + tmp8;
                                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp9);
                                }
                            }
                            tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                            tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer(x2, 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (200704L*x0) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (256L*x2) + (50176L*x1))];
                                auto tmp3 = in_ptr2[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))];
                                auto tmp5 = in_ptr3[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))];
                                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                                tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                            }
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0.mean;
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0.m2;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer((x2 + x2_inner), 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (784L*x3_inner) + (200704L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1) + (200704L*x0)));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1) + (200704L*x0)));
                                auto tmp7 = out_ptr0[static_cast<long>(x2 + x2_inner + (196L*x1) + (784L*x0))];
                                auto tmp10 = out_ptr1[static_cast<long>(x2 + x2_inner + (196L*x1) + (784L*x0))];
                                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3));
                                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x3));
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp4 = tmp2 + tmp3;
                                auto tmp6 = tmp4 + tmp5;
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 - tmp8;
                                auto tmp11 = static_cast<float>(256.0);
                                auto tmp12 = tmp10 / tmp11;
                                auto tmp13 = static_cast<float>(1e-06);
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                auto tmp15 = 1 / std::sqrt(tmp14);
                                auto tmp16 = at::vec::Vectorized<float>(tmp15);
                                auto tmp17 = tmp9 * tmp16;
                                auto tmp19 = tmp17 * tmp18;
                                auto tmp21 = tmp19 + tmp20;
                                tmp21.store(out_ptr2 + static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1) + (200704L*x0)));
                            }
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer(x2, 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (200704L*x0) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                            auto tmp1 = in_ptr1[static_cast<long>(x3 + (256L*x2) + (50176L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))];
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp16 = in_ptr4[static_cast<long>(x3)];
                            auto tmp18 = in_ptr5[static_cast<long>(x3)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp10 = static_cast<float>(256.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                            auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                            auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                            out_ptr2[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))] = tmp19;
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (768L*x2) + (602112L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (25088L*x1) + (200704L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(192L); x4+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(256L + x3 + (32L*x1) + (768L*x4) + (150528L*x2) + (602112L*x0)), static_cast<long>(768L), tmp0, 8);
                                for (long x3_inner = 0; x3_inner < 8; x3_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x3_inner));
                                    auto tmp2 = static_cast<float>(0.42044820762685725);
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 * tmp3;
                                    tmp4.store(out_ptr1 + static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (25088L*x1) + (200704L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(192L); x4<static_cast<long>(196L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x3 + (32L*x1) + (768L*x4) + (150528L*x2) + (602112L*x0)));
                                auto tmp1 = static_cast<float>(0.42044820762685725);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr1[static_cast<long>(x4 + (196L*x3) + (196L*x3_inner) + (6272L*x2) + (25088L*x1) + (200704L*x0))] = tmpbuf[x3_inner]; }
                            }
                        }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x3 + (32L*x1) + (768L*x2) + (602112L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (25088L*x1) + (200704L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (25088L*x2) + (200704L*x0)), static_cast<long>(25088L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (8L*x1) + (8L*x1_inner) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_24 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_ptr0[static_cast<long>((14L*(static_cast<long>(x1) % static_cast<long>(2L))) + (28L*(c10::div_floor_integer(x2, 14L))) + (392L*(c10::div_floor_integer(x1, 2L))) + (784L*x3) + (784L*x3_inner) + (200704L*x0) + (static_cast<long>(x2) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp8 = tmp6 + tmp7;
                            tmp8.store(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                        }
                    }
                }
            }
        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(256.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
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


cpp_fused_clone_convolution_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(3584L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (3584L*x2) + (50176L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (3584L*x2) + (50176L*x1) + (100352L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp2.store(out_ptr0 + static_cast<long>(x3 + (3584L*x1) + (7168L*x2) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_27 = async_compile.cpp('''
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
                       float* out_ptr5,
                       float* out_ptr6)
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(29L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(29L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(28);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp5));
                                auto tmp8 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                                auto tmp12 = static_cast<float>(512.0);
                                auto tmp13 = tmp11 / tmp12;
                                auto tmp14 = static_cast<float>(1e-06);
                                auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                                auto tmp16 = 1 / std::sqrt(tmp15);
                                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                auto tmp18 = tmp10 * tmp17;
                                auto tmp19 = masked_load(in_ptr1 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp20 = tmp18 * tmp19;
                                auto tmp21 = masked_load(in_ptr2 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp22 = tmp20 + tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp6(), to_float_mask(tmp5));
                            tmp23.store(out_ptr2 + static_cast<long>(x3 + (512L*x2) + (14848L*x1) + (430592L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(512L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(1024L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(14848L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp8, 8);
                            float tmp11[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(15360L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp11, 8);
                            float tmp14[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(15872L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp14, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(29696L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp17, 8);
                            float tmp20[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(30208L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp20, 8);
                            float tmp23[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(30720L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)), static_cast<long>(1024L), tmp23, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(tmp11 + static_cast<long>(8L*x1_inner));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(tmp14 + static_cast<long>(8L*x1_inner));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x1_inner));
                                auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                                auto tmp24 = at::vec::Vectorized<float>::loadu(tmp23 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = at::vec::maximum(tmp3, tmp1);
                                auto tmp7 = at::vec::maximum(tmp6, tmp4);
                                auto tmp10 = at::vec::maximum(tmp9, tmp7);
                                auto tmp13 = at::vec::maximum(tmp12, tmp10);
                                auto tmp16 = at::vec::maximum(tmp15, tmp13);
                                auto tmp19 = at::vec::maximum(tmp18, tmp16);
                                auto tmp22 = at::vec::maximum(tmp21, tmp19);
                                auto tmp25 = at::vec::maximum(tmp24, tmp22);
                                tmp25.store(out_ptr3 + static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(8L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(512L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(1024L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(14848L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(15360L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(15872L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(29696L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(30208L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(30720L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            auto tmp8 = at::vec::maximum(tmp7, tmp6);
                            auto tmp10 = at::vec::maximum(tmp9, tmp8);
                            auto tmp12 = at::vec::maximum(tmp11, tmp10);
                            auto tmp14 = at::vec::maximum(tmp13, tmp12);
                            auto tmp16 = at::vec::maximum(tmp15, tmp14);
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (512L*x1)), static_cast<long>(512L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (512L*x1)), static_cast<long>(512L), tmp1, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (100352L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr4 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr5 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                            auto tmp1 = in_ptr3[static_cast<long>(x2 + (512L*x1))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
                        }
                        out_ptr4[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr5[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr3 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner)));
                            auto tmp4 = out_ptr4[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp7 = out_ptr5[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 - tmp5;
                            auto tmp8 = static_cast<float>(512.0);
                            auto tmp9 = tmp7 / tmp8;
                            auto tmp10 = static_cast<float>(1e-06);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = 1 / std::sqrt(tmp11);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp6 * tmp13;
                            auto tmp16 = tmp14 * tmp15;
                            auto tmp18 = tmp16 + tmp17;
                            tmp18.store(out_ptr6 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr3[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr3[static_cast<long>(x2 + (512L*x1))];
                        auto tmp3 = out_ptr4[static_cast<long>(x1 + (196L*x0))];
                        auto tmp5 = out_ptr5[static_cast<long>(x1 + (196L*x0))];
                        auto tmp12 = in_ptr4[static_cast<long>(x2)];
                        auto tmp14 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp6 = static_cast<float>(512.0);
                        auto tmp7 = tmp5 / tmp6;
                        auto tmp8 = static_cast<float>(1e-06);
                        auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                        auto tmp10 = 1 / std::sqrt(tmp9);
                        auto tmp11 = decltype(tmp4)(tmp4 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        out_ptr6[static_cast<long>(x2 + (512L*x1) + (100352L*x0))] = tmp15;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_31 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (512L*x1)), static_cast<long>(512L), tmp1, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)), static_cast<long>(512L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (512L*x1)), static_cast<long>(512L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)), static_cast<long>(512L), tmp4, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (100352L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp4);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp6 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp5 - tmp7;
                            auto tmp10 = static_cast<float>(512.0);
                            auto tmp11 = tmp9 / tmp10;
                            auto tmp12 = static_cast<float>(1e-06);
                            auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                            auto tmp14 = 1 / std::sqrt(tmp13);
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp8 * tmp15;
                            auto tmp18 = tmp16 * tmp17;
                            auto tmp20 = tmp18 + tmp19;
                            tmp20.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp7 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = in_ptr3[static_cast<long>(x2)];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp8 = static_cast<float>(512.0);
                        auto tmp9 = tmp7 / tmp8;
                        auto tmp10 = static_cast<float>(1e-06);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = 1 / std::sqrt(tmp11);
                        auto tmp13 = decltype(tmp6)(tmp6 * tmp12);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        out_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))] = tmp17;
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


cpp_fused_add_native_layer_norm_33 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (512L*x1)), static_cast<long>(512L), tmp1, 8);
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)), static_cast<long>(512L), tmp4, 8);
                            float tmp7[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)), static_cast<long>(512L), tmp7, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x2 + (512L*x1)), static_cast<long>(512L), tmp1, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)), static_cast<long>(512L), tmp4, 8);
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)), static_cast<long>(512L), tmp7, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (196L*x2_inner) + (100352L*x0)));
                                auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x2_inner));
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x2_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(tmp7 + static_cast<long>(8L*x2_inner));
                                auto tmp3 = tmp0 + tmp2;
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp9 = tmp6 + tmp8;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp9);
                            }
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1))];
                            auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp6);
                        }
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = tmp_acc0.m2;
                    }
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp8 = out_ptr0[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp11 = out_ptr1[static_cast<long>(x1 + x1_inner + (196L*x0))];
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp12 = static_cast<float>(512.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = static_cast<float>(1e-06);
                            auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                            auto tmp16 = 1 / std::sqrt(tmp15);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp10 * tmp17;
                            auto tmp20 = tmp18 * tmp19;
                            auto tmp22 = tmp20 + tmp21;
                            tmp22.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp9 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = in_ptr4[static_cast<long>(x2)];
                        auto tmp18 = in_ptr5[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp10 = static_cast<float>(512.0);
                        auto tmp11 = tmp9 / tmp10;
                        auto tmp12 = static_cast<float>(1e-06);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        auto tmp14 = 1 / std::sqrt(tmp13);
                        auto tmp15 = decltype(tmp8)(tmp8 * tmp14);
                        auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        out_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))] = tmp19;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (196L*x2) + (100352L*x0)), static_cast<long>(196L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                            auto tmp3 = tmp1 + tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp9 = tmp7 + tmp8;
                            tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (512L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x2) + (100352L*x0))];
                        auto tmp1 = in_ptr1[static_cast<long>(x2 + (512L*x1))];
                        auto tmp3 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp5 = in_ptr3[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp7 = in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))];
                        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        in_out_ptr0[static_cast<long>(x2 + (512L*x1) + (100352L*x0))] = tmp8;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_39 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_43 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_45 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_51 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_55 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_57 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_63 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_67 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_69 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_75 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_79 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_81 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_87 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_91 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_93 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_99 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_103 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_105 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_111 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_115 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_117 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_121 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_123 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_127 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_129 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
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
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_133 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_native_layer_norm_135 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(512.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_136 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_138 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_139 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(512.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_140 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_141 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(512.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_142 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            auto tmp1 = static_cast<float>(0.42044820762685725);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = static_cast<float>(0.42044820762685725);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 * tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (301056L*x0)));
                        auto tmp1 = static_cast<float>(0.42044820762685725);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (100352L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp3;
                        tmp_acc0 = tmp_acc0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1024L + x3 + (32L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_144 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (6272L*x2) + (100352L*x0)), static_cast<long>(6272L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_145 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(512.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_146 = async_compile.cpp('''
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


cpp_fused_mean_native_layer_norm_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*x2) + (100352L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (196L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = tmp2 - tmp4;
                            auto tmp7 = static_cast<float>(512.0);
                            auto tmp8 = tmp6 / tmp7;
                            auto tmp9 = static_cast<float>(1e-06);
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            auto tmp11 = 1 / std::sqrt(tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp5 * tmp12;
                            auto tmp15 = tmp13 * tmp14;
                            auto tmp17 = tmp15 + tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(arg1_1, (128, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (1, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(arg12_1, (256, ), (1, ))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (1, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, ), (1, ))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (512, ), (1, ))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, ), (1, ))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (512, ), (1, ))
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (384, 128), (128, 1))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (128, 128), (128, 1))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (512, 128), (128, 1))
    assert_size_stride(arg112_1, (512, ), (1, ))
    assert_size_stride(arg113_1, (128, 512), (512, 1))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (384, 128), (128, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (128, 128), (128, 1))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (512, 128), (128, 1))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (128, 512), (512, 1))
    assert_size_stride(arg122_1, (128, ), (1, ))
    assert_size_stride(arg123_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (768, 256), (256, 1))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (256, 256), (256, 1))
    assert_size_stride(arg128_1, (256, ), (1, ))
    assert_size_stride(arg129_1, (1024, 256), (256, 1))
    assert_size_stride(arg130_1, (1024, ), (1, ))
    assert_size_stride(arg131_1, (256, 1024), (1024, 1))
    assert_size_stride(arg132_1, (256, ), (1, ))
    assert_size_stride(arg133_1, (768, 256), (256, 1))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (256, 256), (256, 1))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (1024, 256), (256, 1))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (256, 1024), (1024, 1))
    assert_size_stride(arg140_1, (256, ), (1, ))
    assert_size_stride(arg141_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg142_1, (512, ), (1, ))
    assert_size_stride(arg143_1, (1536, 512), (512, 1))
    assert_size_stride(arg144_1, (1536, ), (1, ))
    assert_size_stride(arg145_1, (512, 512), (512, 1))
    assert_size_stride(arg146_1, (512, ), (1, ))
    assert_size_stride(arg147_1, (2048, 512), (512, 1))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (512, 2048), (2048, 1))
    assert_size_stride(arg150_1, (512, ), (1, ))
    assert_size_stride(arg151_1, (1536, 512), (512, 1))
    assert_size_stride(arg152_1, (1536, ), (1, ))
    assert_size_stride(arg153_1, (512, 512), (512, 1))
    assert_size_stride(arg154_1, (512, ), (1, ))
    assert_size_stride(arg155_1, (2048, 512), (512, 1))
    assert_size_stride(arg156_1, (2048, ), (1, ))
    assert_size_stride(arg157_1, (512, 2048), (2048, 1))
    assert_size_stride(arg158_1, (512, ), (1, ))
    assert_size_stride(arg159_1, (1536, 512), (512, 1))
    assert_size_stride(arg160_1, (1536, ), (1, ))
    assert_size_stride(arg161_1, (512, 512), (512, 1))
    assert_size_stride(arg162_1, (512, ), (1, ))
    assert_size_stride(arg163_1, (2048, 512), (512, 1))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (512, 2048), (2048, 1))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (1536, 512), (512, 1))
    assert_size_stride(arg168_1, (1536, ), (1, ))
    assert_size_stride(arg169_1, (512, 512), (512, 1))
    assert_size_stride(arg170_1, (512, ), (1, ))
    assert_size_stride(arg171_1, (2048, 512), (512, 1))
    assert_size_stride(arg172_1, (2048, ), (1, ))
    assert_size_stride(arg173_1, (512, 2048), (2048, 1))
    assert_size_stride(arg174_1, (512, ), (1, ))
    assert_size_stride(arg175_1, (1536, 512), (512, 1))
    assert_size_stride(arg176_1, (1536, ), (1, ))
    assert_size_stride(arg177_1, (512, 512), (512, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (2048, 512), (512, 1))
    assert_size_stride(arg180_1, (2048, ), (1, ))
    assert_size_stride(arg181_1, (512, 2048), (2048, 1))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (1536, 512), (512, 1))
    assert_size_stride(arg184_1, (1536, ), (1, ))
    assert_size_stride(arg185_1, (512, 512), (512, 1))
    assert_size_stride(arg186_1, (512, ), (1, ))
    assert_size_stride(arg187_1, (2048, 512), (512, 1))
    assert_size_stride(arg188_1, (2048, ), (1, ))
    assert_size_stride(arg189_1, (512, 2048), (2048, 1))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (1536, 512), (512, 1))
    assert_size_stride(arg192_1, (1536, ), (1, ))
    assert_size_stride(arg193_1, (512, 512), (512, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (2048, 512), (512, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (512, 2048), (2048, 1))
    assert_size_stride(arg198_1, (512, ), (1, ))
    assert_size_stride(arg199_1, (1536, 512), (512, 1))
    assert_size_stride(arg200_1, (1536, ), (1, ))
    assert_size_stride(arg201_1, (512, 512), (512, 1))
    assert_size_stride(arg202_1, (512, ), (1, ))
    assert_size_stride(arg203_1, (2048, 512), (512, 1))
    assert_size_stride(arg204_1, (2048, ), (1, ))
    assert_size_stride(arg205_1, (512, 2048), (2048, 1))
    assert_size_stride(arg206_1, (512, ), (1, ))
    assert_size_stride(arg207_1, (1536, 512), (512, 1))
    assert_size_stride(arg208_1, (1536, ), (1, ))
    assert_size_stride(arg209_1, (512, 512), (512, 1))
    assert_size_stride(arg210_1, (512, ), (1, ))
    assert_size_stride(arg211_1, (2048, 512), (512, 1))
    assert_size_stride(arg212_1, (2048, ), (1, ))
    assert_size_stride(arg213_1, (512, 2048), (2048, 1))
    assert_size_stride(arg214_1, (512, ), (1, ))
    assert_size_stride(arg215_1, (1536, 512), (512, 1))
    assert_size_stride(arg216_1, (1536, ), (1, ))
    assert_size_stride(arg217_1, (512, 512), (512, 1))
    assert_size_stride(arg218_1, (512, ), (1, ))
    assert_size_stride(arg219_1, (2048, 512), (512, 1))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (512, 2048), (2048, 1))
    assert_size_stride(arg222_1, (512, ), (1, ))
    assert_size_stride(arg223_1, (1536, 512), (512, 1))
    assert_size_stride(arg224_1, (1536, ), (1, ))
    assert_size_stride(arg225_1, (512, 512), (512, 1))
    assert_size_stride(arg226_1, (512, ), (1, ))
    assert_size_stride(arg227_1, (2048, 512), (512, 1))
    assert_size_stride(arg228_1, (2048, ), (1, ))
    assert_size_stride(arg229_1, (512, 2048), (2048, 1))
    assert_size_stride(arg230_1, (512, ), (1, ))
    assert_size_stride(arg231_1, (1536, 512), (512, 1))
    assert_size_stride(arg232_1, (1536, ), (1, ))
    assert_size_stride(arg233_1, (512, 512), (512, 1))
    assert_size_stride(arg234_1, (512, ), (1, ))
    assert_size_stride(arg235_1, (2048, 512), (512, 1))
    assert_size_stride(arg236_1, (2048, ), (1, ))
    assert_size_stride(arg237_1, (512, 2048), (2048, 1))
    assert_size_stride(arg238_1, (512, ), (1, ))
    assert_size_stride(arg239_1, (1536, 512), (512, 1))
    assert_size_stride(arg240_1, (1536, ), (1, ))
    assert_size_stride(arg241_1, (512, 512), (512, 1))
    assert_size_stride(arg242_1, (512, ), (1, ))
    assert_size_stride(arg243_1, (2048, 512), (512, 1))
    assert_size_stride(arg244_1, (2048, ), (1, ))
    assert_size_stride(arg245_1, (512, 2048), (2048, 1))
    assert_size_stride(arg246_1, (512, ), (1, ))
    assert_size_stride(arg247_1, (1536, 512), (512, 1))
    assert_size_stride(arg248_1, (1536, ), (1, ))
    assert_size_stride(arg249_1, (512, 512), (512, 1))
    assert_size_stride(arg250_1, (512, ), (1, ))
    assert_size_stride(arg251_1, (2048, 512), (512, 1))
    assert_size_stride(arg252_1, (2048, ), (1, ))
    assert_size_stride(arg253_1, (512, 2048), (2048, 1))
    assert_size_stride(arg254_1, (512, ), (1, ))
    assert_size_stride(arg255_1, (1536, 512), (512, 1))
    assert_size_stride(arg256_1, (1536, ), (1, ))
    assert_size_stride(arg257_1, (512, 512), (512, 1))
    assert_size_stride(arg258_1, (512, ), (1, ))
    assert_size_stride(arg259_1, (2048, 512), (512, 1))
    assert_size_stride(arg260_1, (2048, ), (1, ))
    assert_size_stride(arg261_1, (512, 2048), (2048, 1))
    assert_size_stride(arg262_1, (512, ), (1, ))
    assert_size_stride(arg263_1, (1536, 512), (512, 1))
    assert_size_stride(arg264_1, (1536, ), (1, ))
    assert_size_stride(arg265_1, (512, 512), (512, 1))
    assert_size_stride(arg266_1, (512, ), (1, ))
    assert_size_stride(arg267_1, (2048, 512), (512, 1))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (512, 2048), (2048, 1))
    assert_size_stride(arg270_1, (512, ), (1, ))
    assert_size_stride(arg271_1, (1536, 512), (512, 1))
    assert_size_stride(arg272_1, (1536, ), (1, ))
    assert_size_stride(arg273_1, (512, 512), (512, 1))
    assert_size_stride(arg274_1, (512, ), (1, ))
    assert_size_stride(arg275_1, (2048, 512), (512, 1))
    assert_size_stride(arg276_1, (2048, ), (1, ))
    assert_size_stride(arg277_1, (512, 2048), (2048, 1))
    assert_size_stride(arg278_1, (512, ), (1, ))
    assert_size_stride(arg279_1, (1536, 512), (512, 1))
    assert_size_stride(arg280_1, (1536, ), (1, ))
    assert_size_stride(arg281_1, (512, 512), (512, 1))
    assert_size_stride(arg282_1, (512, ), (1, ))
    assert_size_stride(arg283_1, (2048, 512), (512, 1))
    assert_size_stride(arg284_1, (2048, ), (1, ))
    assert_size_stride(arg285_1, (512, 2048), (2048, 1))
    assert_size_stride(arg286_1, (512, ), (1, ))
    assert_size_stride(arg287_1, (1536, 512), (512, 1))
    assert_size_stride(arg288_1, (1536, ), (1, ))
    assert_size_stride(arg289_1, (512, 512), (512, 1))
    assert_size_stride(arg290_1, (512, ), (1, ))
    assert_size_stride(arg291_1, (2048, 512), (512, 1))
    assert_size_stride(arg292_1, (2048, ), (1, ))
    assert_size_stride(arg293_1, (512, 2048), (2048, 1))
    assert_size_stride(arg294_1, (512, ), (1, ))
    assert_size_stride(arg295_1, (1536, 512), (512, 1))
    assert_size_stride(arg296_1, (1536, ), (1, ))
    assert_size_stride(arg297_1, (512, 512), (512, 1))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (2048, 512), (512, 1))
    assert_size_stride(arg300_1, (2048, ), (1, ))
    assert_size_stride(arg301_1, (512, 2048), (2048, 1))
    assert_size_stride(arg302_1, (512, ), (1, ))
    assert_size_stride(arg303_1, (1000, 512), (512, 1))
    assert_size_stride(arg304_1, (1000, ), (1, ))
    assert_size_stride(arg305_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg305_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg105_1
    del arg305_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg106_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del arg106_1
    del buf0
    del buf1
    buf3 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 16, 196, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg1_1
    del arg2_1
    buf7 = empty((25088, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf6, (25088, 128), (128, 1), 0), reinterpret_tensor(arg107_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf7)
    del arg107_1
    del arg108_1
    buf8 = reinterpret_tensor(buf6, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf6  # reuse
    buf9 = empty((8, 4, 16, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_2(c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    buf10 = empty((512, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf8, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf9, (512, 32, 196), (6272, 196, 1), 0), out=buf10)
    buf11 = empty_strided((8, 4, 16, 196, 1), (12544, 3136, 196, 1, 100352), device='cpu', dtype=torch.float32)
    buf12 = reinterpret_tensor(buf10, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), 0); del buf10  # reuse
    buf13 = empty_strided((8, 4, 16, 196, 1), (12544, 3136, 196, 1, 100352), device='cpu', dtype=torch.float32)
    buf14 = buf12; del buf12  # reuse
    buf15 = reinterpret_tensor(buf9, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf9  # reuse
    cpp_fused__softmax_clone_3(c_void_p(buf14.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf8, (512, 196, 32), (6272, 32, 1), 0); del buf8  # reuse
    # Source Nodes: [x_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf14, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf15, (512, 196, 32), (6272, 32, 1), 0), out=buf16)
    buf17 = reinterpret_tensor(buf15, (8, 16, 196, 32, 4), (401408, 25088, 128, 4, 1), 0); del buf15  # reuse
    cpp_fused_clone_4(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = reinterpret_tensor(buf16, (25088, 128), (128, 1), 0); del buf16  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf17, (25088, 128), (128, 1), 0), reinterpret_tensor(arg109_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf18)
    del arg109_1
    del arg110_1
    buf19 = buf4; del buf4  # reuse
    buf20 = buf3; del buf3  # reuse
    buf22 = reinterpret_tensor(buf17, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf17  # reuse
    cpp_fused_add_native_layer_norm_5(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg3_1
    del arg4_1
    buf23 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg112_1, reinterpret_tensor(buf22, (25088, 128), (128, 1), 0), reinterpret_tensor(arg111_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf23)
    del arg111_1
    del arg112_1
    buf24 = reinterpret_tensor(buf23, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf23  # reuse
    cpp_fused_gelu_6(c_void_p(buf24.data_ptr()))
    buf25 = reinterpret_tensor(buf22, (25088, 128), (128, 1), 0); del buf22  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg114_1, reinterpret_tensor(buf24, (25088, 512), (512, 1), 0), reinterpret_tensor(arg113_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf25)
    del arg113_1
    del arg114_1
    buf26 = buf20; del buf20  # reuse
    buf27 = buf19; del buf19  # reuse
    buf29 = empty((8, 16, 196, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_7(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg5_1
    del arg6_1
    buf30 = buf7; del buf7  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg116_1, reinterpret_tensor(buf29, (25088, 128), (128, 1), 0), reinterpret_tensor(arg115_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf30)
    del arg115_1
    del arg116_1
    buf31 = reinterpret_tensor(buf29, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf29  # reuse
    buf32 = empty((8, 4, 16, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_8(c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf14, (512, 196, 196), (38416, 196, 1), 0); del buf14  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf31, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf32, (512, 32, 196), (6272, 196, 1), 0), out=buf33)
    buf34 = buf13; del buf13  # reuse
    buf35 = reinterpret_tensor(buf33, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), 0); del buf33  # reuse
    buf36 = buf11; del buf11  # reuse
    buf37 = buf35; del buf35  # reuse
    buf38 = reinterpret_tensor(buf32, (8, 4, 16, 196, 32), (401408, 100352, 6272, 32, 1), 0); del buf32  # reuse
    cpp_fused__softmax_clone_9(c_void_p(buf37.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()))
    del buf30
    del buf34
    del buf36
    buf39 = reinterpret_tensor(buf31, (512, 196, 32), (6272, 32, 1), 0); del buf31  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf37, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf38, (512, 196, 32), (6272, 32, 1), 0), out=buf39)
    del buf37
    buf40 = reinterpret_tensor(buf38, (8, 16, 196, 32, 4), (401408, 25088, 128, 4, 1), 0); del buf38  # reuse
    cpp_fused_clone_10(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf39, (25088, 128), (128, 1), 0); del buf39  # reuse
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf40, (25088, 128), (128, 1), 0), reinterpret_tensor(arg117_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf41)
    del arg117_1
    del arg118_1
    buf42 = reinterpret_tensor(buf41, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf41  # reuse
    buf43 = buf27; del buf27  # reuse
    buf44 = buf26; del buf26  # reuse
    buf46 = reinterpret_tensor(buf40, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf40  # reuse
    cpp_fused_add_native_layer_norm_11(c_void_p(buf42.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()))
    del arg0_1
    del arg7_1
    del arg8_1
    del buf18
    del buf2
    buf47 = reinterpret_tensor(buf24, (25088, 512), (512, 1), 0); del buf24  # reuse
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf46, (25088, 128), (128, 1), 0), reinterpret_tensor(arg119_1, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf47)
    del arg119_1
    del arg120_1
    buf48 = reinterpret_tensor(buf47, (8, 16, 196, 512), (1605632, 100352, 512, 1), 0); del buf47  # reuse
    cpp_fused_gelu_12(c_void_p(buf48.data_ptr()))
    buf49 = reinterpret_tensor(buf46, (25088, 128), (128, 1), 0); del buf46  # reuse
    # Source Nodes: [x_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf48, (25088, 512), (512, 1), 0), reinterpret_tensor(arg121_1, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf49)
    del arg121_1
    del arg122_1
    del buf48
    buf50 = reinterpret_tensor(buf25, (8, 4, 14, 4, 14, 128), (401408, 100352, 7168, 1792, 128, 1), 0); del buf25  # reuse
    buf51 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_13(c_void_p(buf42.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg123_1
    del buf42
    del buf49
    # Source Nodes: [x_41], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(reinterpret_tensor(buf50, (8, 128, 56, 56), (401408, 1, 7168, 128), 0), buf51, arg124_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf52, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del arg124_1
    del buf50
    del buf51
    buf53 = reinterpret_tensor(buf44, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf44  # reuse
    buf54 = reinterpret_tensor(buf43, (8, 56, 56, 1), (3136, 56, 1, 25088), 0); del buf43  # reuse
    buf56 = empty_strided((8, 256, 57, 57), (831744, 1, 14592, 256), device='cpu', dtype=torch.float32)
    buf57 = empty((8, 256, 28, 28), device='cpu', dtype=torch.float32)
    buf58 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf61 = empty((8, 4, 196, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_14(c_void_p(buf52.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()))
    del arg10_1
    del arg12_1
    del arg13_1
    del arg9_1
    del buf56
    buf62 = empty((6272, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg126_1, reinterpret_tensor(buf61, (6272, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf62)
    del arg125_1
    del arg126_1
    buf63 = reinterpret_tensor(buf61, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf61  # reuse
    buf64 = empty((8, 8, 4, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_15(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    buf65 = empty((256, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf63, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf64, (256, 32, 196), (6272, 196, 1), 0), out=buf65)
    buf66 = empty_strided((8, 8, 4, 196, 1), (6272, 784, 196, 1, 50176), device='cpu', dtype=torch.float32)
    buf67 = reinterpret_tensor(buf65, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), 0); del buf65  # reuse
    buf68 = empty_strided((8, 8, 4, 196, 1), (6272, 784, 196, 1, 50176), device='cpu', dtype=torch.float32)
    buf69 = buf67; del buf67  # reuse
    buf70 = reinterpret_tensor(buf64, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf64  # reuse
    cpp_fused__softmax_clone_16(c_void_p(buf69.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = reinterpret_tensor(buf63, (256, 196, 32), (6272, 32, 1), 0); del buf63  # reuse
    # Source Nodes: [x_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf69, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf70, (256, 196, 32), (6272, 32, 1), 0), out=buf71)
    buf72 = reinterpret_tensor(buf70, (8, 4, 196, 32, 8), (200704, 50176, 256, 8, 1), 0); del buf70  # reuse
    cpp_fused_clone_17(c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf71, (6272, 256), (256, 1), 0); del buf71  # reuse
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf72, (6272, 256), (256, 1), 0), reinterpret_tensor(arg127_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf73)
    del arg127_1
    del arg128_1
    buf74 = buf59; del buf59  # reuse
    buf75 = buf58; del buf58  # reuse
    buf77 = reinterpret_tensor(buf72, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf72  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf57.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()))
    del arg14_1
    del arg15_1
    buf78 = reinterpret_tensor(buf52, (6272, 1024), (1024, 1), 0); del buf52  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf77, (6272, 256), (256, 1), 0), reinterpret_tensor(arg129_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf78)
    del arg129_1
    del arg130_1
    buf79 = reinterpret_tensor(buf78, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf78  # reuse
    cpp_fused_gelu_19(c_void_p(buf79.data_ptr()))
    buf80 = reinterpret_tensor(buf77, (6272, 256), (256, 1), 0); del buf77  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf79, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg131_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf80)
    del arg131_1
    del arg132_1
    buf81 = buf75; del buf75  # reuse
    buf82 = buf74; del buf74  # reuse
    buf84 = empty((8, 4, 196, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_20(c_void_p(buf57.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()))
    del arg16_1
    del arg17_1
    buf85 = buf62; del buf62  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg134_1, reinterpret_tensor(buf84, (6272, 256), (256, 1), 0), reinterpret_tensor(arg133_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf85)
    del arg133_1
    del arg134_1
    buf86 = reinterpret_tensor(buf84, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf84  # reuse
    buf87 = empty((8, 8, 4, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_21(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    buf88 = reinterpret_tensor(buf69, (256, 196, 196), (38416, 196, 1), 0); del buf69  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf86, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf87, (256, 32, 196), (6272, 196, 1), 0), out=buf88)
    buf89 = buf68; del buf68  # reuse
    buf90 = reinterpret_tensor(buf88, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), 0); del buf88  # reuse
    buf91 = buf66; del buf66  # reuse
    buf92 = buf90; del buf90  # reuse
    buf93 = reinterpret_tensor(buf87, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf87  # reuse
    cpp_fused__softmax_clone_22(c_void_p(buf92.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    del buf85
    del buf89
    del buf91
    buf94 = reinterpret_tensor(buf86, (256, 196, 32), (6272, 32, 1), 0); del buf86  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf92, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf93, (256, 196, 32), (6272, 32, 1), 0), out=buf94)
    del buf92
    buf95 = reinterpret_tensor(buf93, (8, 4, 196, 32, 8), (200704, 50176, 256, 8, 1), 0); del buf93  # reuse
    cpp_fused_clone_23(c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    buf96 = reinterpret_tensor(buf94, (6272, 256), (256, 1), 0); del buf94  # reuse
    # Source Nodes: [x_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg136_1, reinterpret_tensor(buf95, (6272, 256), (256, 1), 0), reinterpret_tensor(arg135_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf96)
    del arg135_1
    del arg136_1
    buf97 = reinterpret_tensor(buf96, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf96  # reuse
    buf98 = buf82; del buf82  # reuse
    buf99 = buf81; del buf81  # reuse
    buf101 = reinterpret_tensor(buf95, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf95  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf97.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg11_1
    del arg18_1
    del arg19_1
    del buf57
    del buf73
    buf102 = reinterpret_tensor(buf79, (6272, 1024), (1024, 1), 0); del buf79  # reuse
    # Source Nodes: [x_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf101, (6272, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf102)
    del arg137_1
    del arg138_1
    buf103 = reinterpret_tensor(buf102, (8, 4, 196, 1024), (802816, 200704, 1024, 1), 0); del buf102  # reuse
    cpp_fused_gelu_25(c_void_p(buf103.data_ptr()))
    buf104 = reinterpret_tensor(buf101, (6272, 256), (256, 1), 0); del buf101  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf103, (6272, 1024), (1024, 1), 0), reinterpret_tensor(arg139_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf104)
    del arg139_1
    del arg140_1
    del buf103
    buf105 = reinterpret_tensor(buf80, (8, 2, 14, 2, 14, 256), (200704, 100352, 7168, 3584, 256, 1), 0); del buf80  # reuse
    buf106 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_26(c_void_p(buf97.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del arg141_1
    del buf104
    del buf97
    # Source Nodes: [x_85], Original ATen: [aten.convolution]
    buf107 = extern_kernels.convolution(reinterpret_tensor(buf105, (8, 256, 28, 28), (200704, 1, 7168, 256), 0), buf106, arg142_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf107, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del arg142_1
    del buf105
    del buf106
    buf108 = reinterpret_tensor(buf99, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf99  # reuse
    buf109 = reinterpret_tensor(buf98, (8, 28, 28, 1), (784, 28, 1, 6272), 0); del buf98  # reuse
    buf111 = empty_strided((8, 512, 29, 29), (430592, 1, 14848, 512), device='cpu', dtype=torch.float32)
    buf112 = empty((8, 512, 14, 14), device='cpu', dtype=torch.float32)
    buf113 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf114 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf116 = empty((8, 1, 196, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_27(c_void_p(buf107.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg20_1
    del arg21_1
    del arg23_1
    del arg24_1
    del buf108
    del buf109
    del buf111
    buf117 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf116, (1568, 512), (512, 1), 0), reinterpret_tensor(arg143_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf117)
    del arg143_1
    del arg144_1
    buf118 = reinterpret_tensor(buf116, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf116  # reuse
    buf119 = empty((8, 16, 1, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_28(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_98], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf118, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf119, (128, 32, 196), (6272, 196, 1), 0), out=buf120)
    buf121 = reinterpret_tensor(buf54, (8, 16, 1, 196, 1), (3136, 196, 25088, 1, 25088), 0); del buf54  # reuse
    buf122 = reinterpret_tensor(buf120, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf120  # reuse
    buf123 = reinterpret_tensor(buf53, (8, 16, 1, 196, 1), (3136, 196, 25088, 1, 25088), 0); del buf53  # reuse
    buf124 = reinterpret_tensor(buf122, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf122  # reuse
    buf125 = reinterpret_tensor(buf119, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf119  # reuse
    cpp_fused__softmax_clone_29(c_void_p(buf124.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()))
    buf126 = reinterpret_tensor(buf118, (128, 196, 32), (6272, 32, 1), 0); del buf118  # reuse
    # Source Nodes: [x_98], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf124, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf125, (128, 196, 32), (6272, 32, 1), 0), out=buf126)
    buf127 = reinterpret_tensor(buf125, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf125  # reuse
    cpp_fused_clone_30(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    buf128 = reinterpret_tensor(buf126, (1568, 512), (512, 1), 0); del buf126  # reuse
    # Source Nodes: [x_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg146_1, reinterpret_tensor(buf127, (1568, 512), (512, 1), 0), reinterpret_tensor(arg145_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf128)
    del arg145_1
    del arg146_1
    buf129 = buf114; del buf114  # reuse
    buf130 = buf113; del buf113  # reuse
    buf132 = reinterpret_tensor(buf127, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf127  # reuse
    cpp_fused_add_native_layer_norm_31(c_void_p(buf112.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf132.data_ptr()))
    del arg25_1
    del arg26_1
    buf133 = reinterpret_tensor(buf107, (1568, 2048), (2048, 1), 0); del buf107  # reuse
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf132, (1568, 512), (512, 1), 0), reinterpret_tensor(arg147_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf133)
    del arg147_1
    del arg148_1
    buf134 = reinterpret_tensor(buf133, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf133  # reuse
    cpp_fused_gelu_32(c_void_p(buf134.data_ptr()))
    buf135 = reinterpret_tensor(buf132, (1568, 512), (512, 1), 0); del buf132  # reuse
    # Source Nodes: [x_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf134, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg149_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf135)
    del arg149_1
    del arg150_1
    buf136 = buf130; del buf130  # reuse
    buf137 = buf129; del buf129  # reuse
    buf139 = empty((8, 1, 196, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_33(c_void_p(buf112.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()))
    del arg27_1
    del arg28_1
    buf140 = buf117; del buf117  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf139, (1568, 512), (512, 1), 0), reinterpret_tensor(arg151_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf140)
    del arg151_1
    del arg152_1
    buf141 = reinterpret_tensor(buf139, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf139  # reuse
    buf142 = empty((8, 16, 1, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_34(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    buf143 = reinterpret_tensor(buf124, (128, 196, 196), (38416, 196, 1), 0); del buf124  # reuse
    # Source Nodes: [x_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf141, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf142, (128, 32, 196), (6272, 196, 1), 0), out=buf143)
    buf144 = buf123; del buf123  # reuse
    buf145 = reinterpret_tensor(buf143, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf143  # reuse
    buf146 = buf121; del buf121  # reuse
    buf147 = reinterpret_tensor(buf145, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf145  # reuse
    buf148 = reinterpret_tensor(buf142, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf142  # reuse
    cpp_fused__softmax_clone_35(c_void_p(buf147.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()))
    buf149 = reinterpret_tensor(buf141, (128, 196, 32), (6272, 32, 1), 0); del buf141  # reuse
    # Source Nodes: [x_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf147, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf148, (128, 196, 32), (6272, 32, 1), 0), out=buf149)
    buf150 = reinterpret_tensor(buf148, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf148  # reuse
    cpp_fused_clone_36(c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    buf151 = reinterpret_tensor(buf149, (1568, 512), (512, 1), 0); del buf149  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf150, (1568, 512), (512, 1), 0), reinterpret_tensor(arg153_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf151)
    del arg153_1
    del arg154_1
    buf152 = reinterpret_tensor(buf151, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf151  # reuse
    buf153 = buf137; del buf137  # reuse
    buf154 = buf136; del buf136  # reuse
    buf156 = reinterpret_tensor(buf150, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf150  # reuse
    cpp_fused_add_native_layer_norm_37(c_void_p(buf152.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg22_1
    del arg29_1
    del arg30_1
    buf157 = reinterpret_tensor(buf134, (1568, 2048), (2048, 1), 0); del buf134  # reuse
    # Source Nodes: [x_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf156, (1568, 512), (512, 1), 0), reinterpret_tensor(arg155_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf157)
    del arg155_1
    del arg156_1
    buf158 = reinterpret_tensor(buf157, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf157  # reuse
    cpp_fused_gelu_38(c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf156, (1568, 512), (512, 1), 0); del buf156  # reuse
    # Source Nodes: [x_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg158_1, reinterpret_tensor(buf158, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg157_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf159)
    del arg157_1
    del arg158_1
    buf160 = buf154; del buf154  # reuse
    buf161 = buf153; del buf153  # reuse
    buf163 = reinterpret_tensor(buf135, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf135  # reuse
    cpp_fused_add_native_layer_norm_39(c_void_p(buf152.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg31_1
    del arg32_1
    buf164 = buf140; del buf140  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf163, (1568, 512), (512, 1), 0), reinterpret_tensor(arg159_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf164)
    del arg159_1
    del arg160_1
    buf165 = reinterpret_tensor(buf163, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf163  # reuse
    buf166 = reinterpret_tensor(buf128, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf128  # reuse
    cpp_fused_clone_mul_40(c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = reinterpret_tensor(buf147, (128, 196, 196), (38416, 196, 1), 0); del buf147  # reuse
    # Source Nodes: [x_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf165, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf166, (128, 32, 196), (6272, 196, 1), 0), out=buf167)
    buf168 = buf146; del buf146  # reuse
    buf169 = reinterpret_tensor(buf167, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf167  # reuse
    buf170 = buf144; del buf144  # reuse
    buf171 = reinterpret_tensor(buf169, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf169  # reuse
    buf172 = reinterpret_tensor(buf166, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf166  # reuse
    cpp_fused__softmax_clone_41(c_void_p(buf171.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()))
    buf173 = reinterpret_tensor(buf165, (128, 196, 32), (6272, 32, 1), 0); del buf165  # reuse
    # Source Nodes: [x_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf171, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf172, (128, 196, 32), (6272, 32, 1), 0), out=buf173)
    buf174 = reinterpret_tensor(buf172, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf172  # reuse
    cpp_fused_clone_42(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = reinterpret_tensor(buf173, (1568, 512), (512, 1), 0); del buf173  # reuse
    # Source Nodes: [x_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf174, (1568, 512), (512, 1), 0), reinterpret_tensor(arg161_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf175)
    del arg161_1
    del arg162_1
    buf176 = buf161; del buf161  # reuse
    buf177 = buf160; del buf160  # reuse
    buf179 = reinterpret_tensor(buf174, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf174  # reuse
    cpp_fused_add_native_layer_norm_43(c_void_p(buf152.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()))
    del arg33_1
    del arg34_1
    buf180 = reinterpret_tensor(buf158, (1568, 2048), (2048, 1), 0); del buf158  # reuse
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg164_1, reinterpret_tensor(buf179, (1568, 512), (512, 1), 0), reinterpret_tensor(arg163_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf180)
    del arg163_1
    del arg164_1
    buf181 = reinterpret_tensor(buf180, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf180  # reuse
    cpp_fused_gelu_44(c_void_p(buf181.data_ptr()))
    buf182 = reinterpret_tensor(buf179, (1568, 512), (512, 1), 0); del buf179  # reuse
    # Source Nodes: [x_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf181, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg165_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf182)
    del arg165_1
    del arg166_1
    buf183 = buf177; del buf177  # reuse
    buf184 = buf176; del buf176  # reuse
    buf186 = reinterpret_tensor(buf112, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf112  # reuse
    cpp_fused_add_native_layer_norm_45(c_void_p(buf152.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()))
    del arg35_1
    del arg36_1
    buf187 = buf164; del buf164  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf186, (1568, 512), (512, 1), 0), reinterpret_tensor(arg167_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf187)
    del arg167_1
    del arg168_1
    buf188 = reinterpret_tensor(buf186, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf186  # reuse
    buf189 = empty((8, 16, 1, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_46(c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = reinterpret_tensor(buf171, (128, 196, 196), (38416, 196, 1), 0); del buf171  # reuse
    # Source Nodes: [x_140], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf189, (128, 32, 196), (6272, 196, 1), 0), out=buf190)
    buf191 = buf170; del buf170  # reuse
    buf192 = reinterpret_tensor(buf190, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf190  # reuse
    buf193 = buf168; del buf168  # reuse
    buf194 = reinterpret_tensor(buf192, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf192  # reuse
    buf195 = reinterpret_tensor(buf189, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf189  # reuse
    cpp_fused__softmax_clone_47(c_void_p(buf194.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    buf196 = reinterpret_tensor(buf188, (128, 196, 32), (6272, 32, 1), 0); del buf188  # reuse
    # Source Nodes: [x_140], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf194, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf195, (128, 196, 32), (6272, 32, 1), 0), out=buf196)
    buf197 = reinterpret_tensor(buf195, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf195  # reuse
    cpp_fused_clone_48(c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()))
    buf198 = reinterpret_tensor(buf196, (1568, 512), (512, 1), 0); del buf196  # reuse
    # Source Nodes: [x_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf197, (1568, 512), (512, 1), 0), reinterpret_tensor(arg169_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf198)
    del arg169_1
    del arg170_1
    buf199 = reinterpret_tensor(buf198, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf198  # reuse
    buf200 = buf184; del buf184  # reuse
    buf201 = buf183; del buf183  # reuse
    buf203 = reinterpret_tensor(buf197, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf197  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf199.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()))
    del arg37_1
    del arg38_1
    buf204 = reinterpret_tensor(buf181, (1568, 2048), (2048, 1), 0); del buf181  # reuse
    # Source Nodes: [x_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf203, (1568, 512), (512, 1), 0), reinterpret_tensor(arg171_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf204)
    del arg171_1
    del arg172_1
    buf205 = reinterpret_tensor(buf204, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf204  # reuse
    cpp_fused_gelu_50(c_void_p(buf205.data_ptr()))
    buf206 = reinterpret_tensor(buf203, (1568, 512), (512, 1), 0); del buf203  # reuse
    # Source Nodes: [x_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf205, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg173_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf206)
    del arg173_1
    del arg174_1
    buf207 = buf201; del buf201  # reuse
    buf208 = buf200; del buf200  # reuse
    buf210 = reinterpret_tensor(buf182, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf182  # reuse
    cpp_fused_add_native_layer_norm_51(c_void_p(buf199.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()))
    del arg39_1
    del arg40_1
    buf211 = buf187; del buf187  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf210, (1568, 512), (512, 1), 0), reinterpret_tensor(arg175_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf211)
    del arg175_1
    del arg176_1
    buf212 = reinterpret_tensor(buf210, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf210  # reuse
    buf213 = reinterpret_tensor(buf175, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf175  # reuse
    cpp_fused_clone_mul_52(c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = reinterpret_tensor(buf194, (128, 196, 196), (38416, 196, 1), 0); del buf194  # reuse
    # Source Nodes: [x_154], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf212, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf213, (128, 32, 196), (6272, 196, 1), 0), out=buf214)
    buf215 = buf193; del buf193  # reuse
    buf216 = reinterpret_tensor(buf214, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf214  # reuse
    buf217 = buf191; del buf191  # reuse
    buf218 = reinterpret_tensor(buf216, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf216  # reuse
    buf219 = reinterpret_tensor(buf213, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf213  # reuse
    cpp_fused__softmax_clone_53(c_void_p(buf218.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = reinterpret_tensor(buf212, (128, 196, 32), (6272, 32, 1), 0); del buf212  # reuse
    # Source Nodes: [x_154], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf218, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf219, (128, 196, 32), (6272, 32, 1), 0), out=buf220)
    buf221 = reinterpret_tensor(buf219, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf219  # reuse
    cpp_fused_clone_54(c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = reinterpret_tensor(buf220, (1568, 512), (512, 1), 0); del buf220  # reuse
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf221, (1568, 512), (512, 1), 0), reinterpret_tensor(arg177_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf222)
    del arg177_1
    del arg178_1
    buf223 = buf208; del buf208  # reuse
    buf224 = buf207; del buf207  # reuse
    buf226 = reinterpret_tensor(buf221, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf221  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf199.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()))
    del arg41_1
    del arg42_1
    buf227 = reinterpret_tensor(buf205, (1568, 2048), (2048, 1), 0); del buf205  # reuse
    # Source Nodes: [x_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg180_1, reinterpret_tensor(buf226, (1568, 512), (512, 1), 0), reinterpret_tensor(arg179_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf227)
    del arg179_1
    del arg180_1
    buf228 = reinterpret_tensor(buf227, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf227  # reuse
    cpp_fused_gelu_56(c_void_p(buf228.data_ptr()))
    buf229 = reinterpret_tensor(buf226, (1568, 512), (512, 1), 0); del buf226  # reuse
    # Source Nodes: [x_164], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg182_1, reinterpret_tensor(buf228, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg181_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf229)
    del arg181_1
    del arg182_1
    buf230 = buf224; del buf224  # reuse
    buf231 = buf223; del buf223  # reuse
    buf233 = reinterpret_tensor(buf159, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf159  # reuse
    cpp_fused_add_native_layer_norm_57(c_void_p(buf199.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg43_1
    del arg44_1
    buf234 = buf211; del buf211  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf233, (1568, 512), (512, 1), 0), reinterpret_tensor(arg183_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf234)
    del arg183_1
    del arg184_1
    buf235 = reinterpret_tensor(buf233, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf233  # reuse
    buf236 = reinterpret_tensor(buf152, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf152  # reuse
    cpp_fused_clone_mul_58(c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf218, (128, 196, 196), (38416, 196, 1), 0); del buf218  # reuse
    # Source Nodes: [x_168], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf235, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf236, (128, 32, 196), (6272, 196, 1), 0), out=buf237)
    buf238 = buf217; del buf217  # reuse
    buf239 = reinterpret_tensor(buf237, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf237  # reuse
    buf240 = buf215; del buf215  # reuse
    buf241 = reinterpret_tensor(buf239, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf239  # reuse
    buf242 = reinterpret_tensor(buf236, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf236  # reuse
    cpp_fused__softmax_clone_59(c_void_p(buf241.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf235, (128, 196, 32), (6272, 32, 1), 0); del buf235  # reuse
    # Source Nodes: [x_168], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf241, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf242, (128, 196, 32), (6272, 32, 1), 0), out=buf243)
    buf244 = reinterpret_tensor(buf242, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf242  # reuse
    cpp_fused_clone_60(c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()))
    buf245 = reinterpret_tensor(buf243, (1568, 512), (512, 1), 0); del buf243  # reuse
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg186_1, reinterpret_tensor(buf244, (1568, 512), (512, 1), 0), reinterpret_tensor(arg185_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf245)
    del arg185_1
    del arg186_1
    buf246 = reinterpret_tensor(buf245, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf245  # reuse
    buf247 = buf231; del buf231  # reuse
    buf248 = buf230; del buf230  # reuse
    buf250 = reinterpret_tensor(buf244, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf244  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf246.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()))
    del arg45_1
    del arg46_1
    buf251 = reinterpret_tensor(buf228, (1568, 2048), (2048, 1), 0); del buf228  # reuse
    # Source Nodes: [x_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf250, (1568, 512), (512, 1), 0), reinterpret_tensor(arg187_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf251)
    del arg187_1
    del arg188_1
    buf252 = reinterpret_tensor(buf251, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf251  # reuse
    cpp_fused_gelu_62(c_void_p(buf252.data_ptr()))
    buf253 = reinterpret_tensor(buf250, (1568, 512), (512, 1), 0); del buf250  # reuse
    # Source Nodes: [x_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf252, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg189_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf253)
    del arg189_1
    del arg190_1
    buf254 = buf248; del buf248  # reuse
    buf255 = buf247; del buf247  # reuse
    buf257 = reinterpret_tensor(buf229, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf229  # reuse
    cpp_fused_add_native_layer_norm_63(c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del arg47_1
    del arg48_1
    buf258 = buf234; del buf234  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg192_1, reinterpret_tensor(buf257, (1568, 512), (512, 1), 0), reinterpret_tensor(arg191_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf258)
    del arg191_1
    del arg192_1
    buf259 = reinterpret_tensor(buf257, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf257  # reuse
    buf260 = reinterpret_tensor(buf222, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf222  # reuse
    cpp_fused_clone_mul_64(c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = reinterpret_tensor(buf241, (128, 196, 196), (38416, 196, 1), 0); del buf241  # reuse
    # Source Nodes: [x_182], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf260, (128, 32, 196), (6272, 196, 1), 0), out=buf261)
    buf262 = buf240; del buf240  # reuse
    buf263 = reinterpret_tensor(buf261, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf261  # reuse
    buf264 = buf238; del buf238  # reuse
    buf265 = reinterpret_tensor(buf263, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf263  # reuse
    buf266 = reinterpret_tensor(buf260, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf260  # reuse
    cpp_fused__softmax_clone_65(c_void_p(buf265.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf259, (128, 196, 32), (6272, 32, 1), 0); del buf259  # reuse
    # Source Nodes: [x_182], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf265, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf266, (128, 196, 32), (6272, 32, 1), 0), out=buf267)
    buf268 = reinterpret_tensor(buf266, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf266  # reuse
    cpp_fused_clone_66(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf267, (1568, 512), (512, 1), 0); del buf267  # reuse
    # Source Nodes: [x_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf268, (1568, 512), (512, 1), 0), reinterpret_tensor(arg193_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf269)
    del arg193_1
    del arg194_1
    buf270 = buf255; del buf255  # reuse
    buf271 = buf254; del buf254  # reuse
    buf273 = reinterpret_tensor(buf268, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf268  # reuse
    cpp_fused_add_native_layer_norm_67(c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()))
    del arg49_1
    del arg50_1
    buf274 = reinterpret_tensor(buf252, (1568, 2048), (2048, 1), 0); del buf252  # reuse
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf273, (1568, 512), (512, 1), 0), reinterpret_tensor(arg195_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf274)
    del arg195_1
    del arg196_1
    buf275 = reinterpret_tensor(buf274, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf274  # reuse
    cpp_fused_gelu_68(c_void_p(buf275.data_ptr()))
    buf276 = reinterpret_tensor(buf273, (1568, 512), (512, 1), 0); del buf273  # reuse
    # Source Nodes: [x_192], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf275, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg197_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf276)
    del arg197_1
    del arg198_1
    buf277 = buf271; del buf271  # reuse
    buf278 = buf270; del buf270  # reuse
    buf280 = reinterpret_tensor(buf206, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf206  # reuse
    cpp_fused_add_native_layer_norm_69(c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()))
    del arg51_1
    del arg52_1
    buf281 = buf258; del buf258  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg200_1, reinterpret_tensor(buf280, (1568, 512), (512, 1), 0), reinterpret_tensor(arg199_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf281)
    del arg199_1
    del arg200_1
    buf282 = reinterpret_tensor(buf280, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf280  # reuse
    buf283 = reinterpret_tensor(buf199, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf199  # reuse
    cpp_fused_clone_mul_70(c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    buf284 = reinterpret_tensor(buf265, (128, 196, 196), (38416, 196, 1), 0); del buf265  # reuse
    # Source Nodes: [x_196], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf282, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf283, (128, 32, 196), (6272, 196, 1), 0), out=buf284)
    buf285 = buf264; del buf264  # reuse
    buf286 = reinterpret_tensor(buf284, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf284  # reuse
    buf287 = buf262; del buf262  # reuse
    buf288 = reinterpret_tensor(buf286, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf286  # reuse
    buf289 = reinterpret_tensor(buf283, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf283  # reuse
    cpp_fused__softmax_clone_71(c_void_p(buf288.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf282, (128, 196, 32), (6272, 32, 1), 0); del buf282  # reuse
    # Source Nodes: [x_196], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf288, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf289, (128, 196, 32), (6272, 32, 1), 0), out=buf290)
    buf291 = reinterpret_tensor(buf289, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf289  # reuse
    cpp_fused_clone_72(c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    buf292 = reinterpret_tensor(buf290, (1568, 512), (512, 1), 0); del buf290  # reuse
    # Source Nodes: [x_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg202_1, reinterpret_tensor(buf291, (1568, 512), (512, 1), 0), reinterpret_tensor(arg201_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf292)
    del arg201_1
    del arg202_1
    buf293 = reinterpret_tensor(buf292, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf292  # reuse
    buf294 = buf278; del buf278  # reuse
    buf295 = buf277; del buf277  # reuse
    buf297 = reinterpret_tensor(buf291, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf291  # reuse
    cpp_fused_add_native_layer_norm_73(c_void_p(buf293.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()))
    del arg53_1
    del arg54_1
    buf298 = reinterpret_tensor(buf275, (1568, 2048), (2048, 1), 0); del buf275  # reuse
    # Source Nodes: [x_202], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg204_1, reinterpret_tensor(buf297, (1568, 512), (512, 1), 0), reinterpret_tensor(arg203_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf298)
    del arg203_1
    del arg204_1
    buf299 = reinterpret_tensor(buf298, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf298  # reuse
    cpp_fused_gelu_74(c_void_p(buf299.data_ptr()))
    buf300 = reinterpret_tensor(buf297, (1568, 512), (512, 1), 0); del buf297  # reuse
    # Source Nodes: [x_206], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg206_1, reinterpret_tensor(buf299, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg205_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf300)
    del arg205_1
    del arg206_1
    buf301 = buf295; del buf295  # reuse
    buf302 = buf294; del buf294  # reuse
    buf304 = reinterpret_tensor(buf276, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf276  # reuse
    cpp_fused_add_native_layer_norm_75(c_void_p(buf293.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()))
    del arg55_1
    del arg56_1
    buf305 = buf281; del buf281  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg208_1, reinterpret_tensor(buf304, (1568, 512), (512, 1), 0), reinterpret_tensor(arg207_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf305)
    del arg207_1
    del arg208_1
    buf306 = reinterpret_tensor(buf304, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf304  # reuse
    buf307 = reinterpret_tensor(buf269, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf269  # reuse
    cpp_fused_clone_mul_76(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    buf308 = reinterpret_tensor(buf288, (128, 196, 196), (38416, 196, 1), 0); del buf288  # reuse
    # Source Nodes: [x_210], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf306, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf307, (128, 32, 196), (6272, 196, 1), 0), out=buf308)
    buf309 = buf287; del buf287  # reuse
    buf310 = reinterpret_tensor(buf308, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf308  # reuse
    buf311 = buf285; del buf285  # reuse
    buf312 = reinterpret_tensor(buf310, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf310  # reuse
    buf313 = reinterpret_tensor(buf307, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf307  # reuse
    cpp_fused__softmax_clone_77(c_void_p(buf312.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = reinterpret_tensor(buf306, (128, 196, 32), (6272, 32, 1), 0); del buf306  # reuse
    # Source Nodes: [x_210], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf312, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf313, (128, 196, 32), (6272, 32, 1), 0), out=buf314)
    buf315 = reinterpret_tensor(buf313, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf313  # reuse
    cpp_fused_clone_78(c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    buf316 = reinterpret_tensor(buf314, (1568, 512), (512, 1), 0); del buf314  # reuse
    # Source Nodes: [x_212], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg210_1, reinterpret_tensor(buf315, (1568, 512), (512, 1), 0), reinterpret_tensor(arg209_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf316)
    del arg209_1
    del arg210_1
    buf317 = buf302; del buf302  # reuse
    buf318 = buf301; del buf301  # reuse
    buf320 = reinterpret_tensor(buf315, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf315  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf293.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()))
    del arg57_1
    del arg58_1
    buf321 = reinterpret_tensor(buf299, (1568, 2048), (2048, 1), 0); del buf299  # reuse
    # Source Nodes: [x_216], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg212_1, reinterpret_tensor(buf320, (1568, 512), (512, 1), 0), reinterpret_tensor(arg211_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf321)
    del arg211_1
    del arg212_1
    buf322 = reinterpret_tensor(buf321, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf321  # reuse
    cpp_fused_gelu_80(c_void_p(buf322.data_ptr()))
    buf323 = reinterpret_tensor(buf320, (1568, 512), (512, 1), 0); del buf320  # reuse
    # Source Nodes: [x_220], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg214_1, reinterpret_tensor(buf322, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg213_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf323)
    del arg213_1
    del arg214_1
    buf324 = buf318; del buf318  # reuse
    buf325 = buf317; del buf317  # reuse
    buf327 = reinterpret_tensor(buf253, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf253  # reuse
    cpp_fused_add_native_layer_norm_81(c_void_p(buf293.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()))
    del arg59_1
    del arg60_1
    buf328 = buf305; del buf305  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg216_1, reinterpret_tensor(buf327, (1568, 512), (512, 1), 0), reinterpret_tensor(arg215_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf328)
    del arg215_1
    del arg216_1
    buf329 = reinterpret_tensor(buf327, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf327  # reuse
    buf330 = reinterpret_tensor(buf246, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf246  # reuse
    cpp_fused_clone_mul_82(c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    buf331 = reinterpret_tensor(buf312, (128, 196, 196), (38416, 196, 1), 0); del buf312  # reuse
    # Source Nodes: [x_224], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf329, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf330, (128, 32, 196), (6272, 196, 1), 0), out=buf331)
    buf332 = buf311; del buf311  # reuse
    buf333 = reinterpret_tensor(buf331, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf331  # reuse
    buf334 = buf309; del buf309  # reuse
    buf335 = reinterpret_tensor(buf333, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf333  # reuse
    buf336 = reinterpret_tensor(buf330, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf330  # reuse
    cpp_fused__softmax_clone_83(c_void_p(buf335.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = reinterpret_tensor(buf329, (128, 196, 32), (6272, 32, 1), 0); del buf329  # reuse
    # Source Nodes: [x_224], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf335, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf336, (128, 196, 32), (6272, 32, 1), 0), out=buf337)
    buf338 = reinterpret_tensor(buf336, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf336  # reuse
    cpp_fused_clone_84(c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    buf339 = reinterpret_tensor(buf337, (1568, 512), (512, 1), 0); del buf337  # reuse
    # Source Nodes: [x_226], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf338, (1568, 512), (512, 1), 0), reinterpret_tensor(arg217_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf339)
    del arg217_1
    del arg218_1
    buf340 = reinterpret_tensor(buf339, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf339  # reuse
    buf341 = buf325; del buf325  # reuse
    buf342 = buf324; del buf324  # reuse
    buf344 = reinterpret_tensor(buf338, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf338  # reuse
    cpp_fused_add_native_layer_norm_85(c_void_p(buf340.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()))
    del arg61_1
    del arg62_1
    buf345 = reinterpret_tensor(buf322, (1568, 2048), (2048, 1), 0); del buf322  # reuse
    # Source Nodes: [x_230], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf344, (1568, 512), (512, 1), 0), reinterpret_tensor(arg219_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf345)
    del arg219_1
    del arg220_1
    buf346 = reinterpret_tensor(buf345, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf345  # reuse
    cpp_fused_gelu_86(c_void_p(buf346.data_ptr()))
    buf347 = reinterpret_tensor(buf344, (1568, 512), (512, 1), 0); del buf344  # reuse
    # Source Nodes: [x_234], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg222_1, reinterpret_tensor(buf346, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg221_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf347)
    del arg221_1
    del arg222_1
    buf348 = buf342; del buf342  # reuse
    buf349 = buf341; del buf341  # reuse
    buf351 = reinterpret_tensor(buf323, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf323  # reuse
    cpp_fused_add_native_layer_norm_87(c_void_p(buf340.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()))
    del arg63_1
    del arg64_1
    buf352 = buf328; del buf328  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg224_1, reinterpret_tensor(buf351, (1568, 512), (512, 1), 0), reinterpret_tensor(arg223_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf352)
    del arg223_1
    del arg224_1
    buf353 = reinterpret_tensor(buf351, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf351  # reuse
    buf354 = reinterpret_tensor(buf316, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf316  # reuse
    cpp_fused_clone_mul_88(c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    buf355 = reinterpret_tensor(buf335, (128, 196, 196), (38416, 196, 1), 0); del buf335  # reuse
    # Source Nodes: [x_238], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf353, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf354, (128, 32, 196), (6272, 196, 1), 0), out=buf355)
    buf356 = buf334; del buf334  # reuse
    buf357 = reinterpret_tensor(buf355, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf355  # reuse
    buf358 = buf332; del buf332  # reuse
    buf359 = reinterpret_tensor(buf357, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf357  # reuse
    buf360 = reinterpret_tensor(buf354, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf354  # reuse
    cpp_fused__softmax_clone_89(c_void_p(buf359.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()))
    buf361 = reinterpret_tensor(buf353, (128, 196, 32), (6272, 32, 1), 0); del buf353  # reuse
    # Source Nodes: [x_238], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf359, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf360, (128, 196, 32), (6272, 32, 1), 0), out=buf361)
    buf362 = reinterpret_tensor(buf360, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf360  # reuse
    cpp_fused_clone_90(c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = reinterpret_tensor(buf361, (1568, 512), (512, 1), 0); del buf361  # reuse
    # Source Nodes: [x_240], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg226_1, reinterpret_tensor(buf362, (1568, 512), (512, 1), 0), reinterpret_tensor(arg225_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf363)
    del arg225_1
    del arg226_1
    buf364 = buf349; del buf349  # reuse
    buf365 = buf348; del buf348  # reuse
    buf367 = reinterpret_tensor(buf362, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf362  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf340.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()))
    del arg65_1
    del arg66_1
    buf368 = reinterpret_tensor(buf346, (1568, 2048), (2048, 1), 0); del buf346  # reuse
    # Source Nodes: [x_244], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg228_1, reinterpret_tensor(buf367, (1568, 512), (512, 1), 0), reinterpret_tensor(arg227_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf368)
    del arg227_1
    del arg228_1
    buf369 = reinterpret_tensor(buf368, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf368  # reuse
    cpp_fused_gelu_92(c_void_p(buf369.data_ptr()))
    buf370 = reinterpret_tensor(buf367, (1568, 512), (512, 1), 0); del buf367  # reuse
    # Source Nodes: [x_248], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg230_1, reinterpret_tensor(buf369, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg229_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf370)
    del arg229_1
    del arg230_1
    buf371 = buf365; del buf365  # reuse
    buf372 = buf364; del buf364  # reuse
    buf374 = reinterpret_tensor(buf300, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf300  # reuse
    cpp_fused_add_native_layer_norm_93(c_void_p(buf340.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf374.data_ptr()))
    del arg67_1
    del arg68_1
    buf375 = buf352; del buf352  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg232_1, reinterpret_tensor(buf374, (1568, 512), (512, 1), 0), reinterpret_tensor(arg231_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf375)
    del arg231_1
    del arg232_1
    buf376 = reinterpret_tensor(buf374, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf374  # reuse
    buf377 = reinterpret_tensor(buf293, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf293  # reuse
    cpp_fused_clone_mul_94(c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    buf378 = reinterpret_tensor(buf359, (128, 196, 196), (38416, 196, 1), 0); del buf359  # reuse
    # Source Nodes: [x_252], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf376, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf377, (128, 32, 196), (6272, 196, 1), 0), out=buf378)
    buf379 = buf358; del buf358  # reuse
    buf380 = reinterpret_tensor(buf378, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf378  # reuse
    buf381 = buf356; del buf356  # reuse
    buf382 = reinterpret_tensor(buf380, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf380  # reuse
    buf383 = reinterpret_tensor(buf377, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf377  # reuse
    cpp_fused__softmax_clone_95(c_void_p(buf382.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf383.data_ptr()))
    buf384 = reinterpret_tensor(buf376, (128, 196, 32), (6272, 32, 1), 0); del buf376  # reuse
    # Source Nodes: [x_252], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf382, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf383, (128, 196, 32), (6272, 32, 1), 0), out=buf384)
    buf385 = reinterpret_tensor(buf383, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf383  # reuse
    cpp_fused_clone_96(c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()))
    buf386 = reinterpret_tensor(buf384, (1568, 512), (512, 1), 0); del buf384  # reuse
    # Source Nodes: [x_254], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg234_1, reinterpret_tensor(buf385, (1568, 512), (512, 1), 0), reinterpret_tensor(arg233_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf386)
    del arg233_1
    del arg234_1
    buf387 = reinterpret_tensor(buf386, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf386  # reuse
    buf388 = buf372; del buf372  # reuse
    buf389 = buf371; del buf371  # reuse
    buf391 = reinterpret_tensor(buf385, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf385  # reuse
    cpp_fused_add_native_layer_norm_97(c_void_p(buf387.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()))
    del arg69_1
    del arg70_1
    buf392 = reinterpret_tensor(buf369, (1568, 2048), (2048, 1), 0); del buf369  # reuse
    # Source Nodes: [x_258], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg236_1, reinterpret_tensor(buf391, (1568, 512), (512, 1), 0), reinterpret_tensor(arg235_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf392)
    del arg235_1
    del arg236_1
    buf393 = reinterpret_tensor(buf392, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf392  # reuse
    cpp_fused_gelu_98(c_void_p(buf393.data_ptr()))
    buf394 = reinterpret_tensor(buf391, (1568, 512), (512, 1), 0); del buf391  # reuse
    # Source Nodes: [x_262], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg238_1, reinterpret_tensor(buf393, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg237_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf394)
    del arg237_1
    del arg238_1
    buf395 = buf389; del buf389  # reuse
    buf396 = buf388; del buf388  # reuse
    buf398 = reinterpret_tensor(buf370, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf370  # reuse
    cpp_fused_add_native_layer_norm_99(c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf398.data_ptr()))
    del arg71_1
    del arg72_1
    buf399 = buf375; del buf375  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg240_1, reinterpret_tensor(buf398, (1568, 512), (512, 1), 0), reinterpret_tensor(arg239_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf399)
    del arg239_1
    del arg240_1
    buf400 = reinterpret_tensor(buf398, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf398  # reuse
    buf401 = reinterpret_tensor(buf363, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf363  # reuse
    cpp_fused_clone_mul_100(c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    buf402 = reinterpret_tensor(buf382, (128, 196, 196), (38416, 196, 1), 0); del buf382  # reuse
    # Source Nodes: [x_266], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf400, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf401, (128, 32, 196), (6272, 196, 1), 0), out=buf402)
    buf403 = buf381; del buf381  # reuse
    buf404 = reinterpret_tensor(buf402, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf402  # reuse
    buf405 = buf379; del buf379  # reuse
    buf406 = reinterpret_tensor(buf404, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf404  # reuse
    buf407 = reinterpret_tensor(buf401, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf401  # reuse
    cpp_fused__softmax_clone_101(c_void_p(buf406.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = reinterpret_tensor(buf400, (128, 196, 32), (6272, 32, 1), 0); del buf400  # reuse
    # Source Nodes: [x_266], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf406, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf407, (128, 196, 32), (6272, 32, 1), 0), out=buf408)
    buf409 = reinterpret_tensor(buf407, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf407  # reuse
    cpp_fused_clone_102(c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = reinterpret_tensor(buf408, (1568, 512), (512, 1), 0); del buf408  # reuse
    # Source Nodes: [x_268], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg242_1, reinterpret_tensor(buf409, (1568, 512), (512, 1), 0), reinterpret_tensor(arg241_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf410)
    del arg241_1
    del arg242_1
    buf411 = buf396; del buf396  # reuse
    buf412 = buf395; del buf395  # reuse
    buf414 = reinterpret_tensor(buf409, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf409  # reuse
    cpp_fused_add_native_layer_norm_103(c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf414.data_ptr()))
    del arg73_1
    del arg74_1
    buf415 = reinterpret_tensor(buf393, (1568, 2048), (2048, 1), 0); del buf393  # reuse
    # Source Nodes: [x_272], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg244_1, reinterpret_tensor(buf414, (1568, 512), (512, 1), 0), reinterpret_tensor(arg243_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf415)
    del arg243_1
    del arg244_1
    buf416 = reinterpret_tensor(buf415, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf415  # reuse
    cpp_fused_gelu_104(c_void_p(buf416.data_ptr()))
    buf417 = reinterpret_tensor(buf414, (1568, 512), (512, 1), 0); del buf414  # reuse
    # Source Nodes: [x_276], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg246_1, reinterpret_tensor(buf416, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg245_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf417)
    del arg245_1
    del arg246_1
    buf418 = buf412; del buf412  # reuse
    buf419 = buf411; del buf411  # reuse
    buf421 = reinterpret_tensor(buf347, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf347  # reuse
    cpp_fused_add_native_layer_norm_105(c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    del arg75_1
    del arg76_1
    buf422 = buf399; del buf399  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg248_1, reinterpret_tensor(buf421, (1568, 512), (512, 1), 0), reinterpret_tensor(arg247_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf422)
    del arg247_1
    del arg248_1
    buf423 = reinterpret_tensor(buf421, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf421  # reuse
    buf424 = reinterpret_tensor(buf340, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf340  # reuse
    cpp_fused_clone_mul_106(c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()))
    buf425 = reinterpret_tensor(buf406, (128, 196, 196), (38416, 196, 1), 0); del buf406  # reuse
    # Source Nodes: [x_280], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf423, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf424, (128, 32, 196), (6272, 196, 1), 0), out=buf425)
    buf426 = buf405; del buf405  # reuse
    buf427 = reinterpret_tensor(buf425, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf425  # reuse
    buf428 = buf403; del buf403  # reuse
    buf429 = reinterpret_tensor(buf427, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf427  # reuse
    buf430 = reinterpret_tensor(buf424, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf424  # reuse
    cpp_fused__softmax_clone_107(c_void_p(buf429.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf430.data_ptr()))
    buf431 = reinterpret_tensor(buf423, (128, 196, 32), (6272, 32, 1), 0); del buf423  # reuse
    # Source Nodes: [x_280], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf429, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf430, (128, 196, 32), (6272, 32, 1), 0), out=buf431)
    buf432 = reinterpret_tensor(buf430, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf430  # reuse
    cpp_fused_clone_108(c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()))
    buf433 = reinterpret_tensor(buf431, (1568, 512), (512, 1), 0); del buf431  # reuse
    # Source Nodes: [x_282], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg250_1, reinterpret_tensor(buf432, (1568, 512), (512, 1), 0), reinterpret_tensor(arg249_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf433)
    del arg249_1
    del arg250_1
    buf434 = reinterpret_tensor(buf433, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf433  # reuse
    buf435 = buf419; del buf419  # reuse
    buf436 = buf418; del buf418  # reuse
    buf438 = reinterpret_tensor(buf432, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf432  # reuse
    cpp_fused_add_native_layer_norm_109(c_void_p(buf434.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf438.data_ptr()))
    del arg77_1
    del arg78_1
    buf439 = reinterpret_tensor(buf416, (1568, 2048), (2048, 1), 0); del buf416  # reuse
    # Source Nodes: [x_286], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg252_1, reinterpret_tensor(buf438, (1568, 512), (512, 1), 0), reinterpret_tensor(arg251_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf439)
    del arg251_1
    del arg252_1
    buf440 = reinterpret_tensor(buf439, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf439  # reuse
    cpp_fused_gelu_110(c_void_p(buf440.data_ptr()))
    buf441 = reinterpret_tensor(buf438, (1568, 512), (512, 1), 0); del buf438  # reuse
    # Source Nodes: [x_290], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg254_1, reinterpret_tensor(buf440, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg253_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf441)
    del arg253_1
    del arg254_1
    buf442 = buf436; del buf436  # reuse
    buf443 = buf435; del buf435  # reuse
    buf445 = reinterpret_tensor(buf417, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf417  # reuse
    cpp_fused_add_native_layer_norm_111(c_void_p(buf434.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf445.data_ptr()))
    del arg79_1
    del arg80_1
    buf446 = buf422; del buf422  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg256_1, reinterpret_tensor(buf445, (1568, 512), (512, 1), 0), reinterpret_tensor(arg255_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf446)
    del arg255_1
    del arg256_1
    buf447 = reinterpret_tensor(buf445, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf445  # reuse
    buf448 = reinterpret_tensor(buf410, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf410  # reuse
    cpp_fused_clone_mul_112(c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = reinterpret_tensor(buf429, (128, 196, 196), (38416, 196, 1), 0); del buf429  # reuse
    # Source Nodes: [x_294], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf447, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf448, (128, 32, 196), (6272, 196, 1), 0), out=buf449)
    buf450 = buf428; del buf428  # reuse
    buf451 = reinterpret_tensor(buf449, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf449  # reuse
    buf452 = buf426; del buf426  # reuse
    buf453 = reinterpret_tensor(buf451, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf451  # reuse
    buf454 = reinterpret_tensor(buf448, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf448  # reuse
    cpp_fused__softmax_clone_113(c_void_p(buf453.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()))
    buf455 = reinterpret_tensor(buf447, (128, 196, 32), (6272, 32, 1), 0); del buf447  # reuse
    # Source Nodes: [x_294], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf453, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf454, (128, 196, 32), (6272, 32, 1), 0), out=buf455)
    buf456 = reinterpret_tensor(buf454, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf454  # reuse
    cpp_fused_clone_114(c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    buf457 = reinterpret_tensor(buf455, (1568, 512), (512, 1), 0); del buf455  # reuse
    # Source Nodes: [x_296], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg258_1, reinterpret_tensor(buf456, (1568, 512), (512, 1), 0), reinterpret_tensor(arg257_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf457)
    del arg257_1
    del arg258_1
    buf458 = buf443; del buf443  # reuse
    buf459 = buf442; del buf442  # reuse
    buf461 = reinterpret_tensor(buf456, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf456  # reuse
    cpp_fused_add_native_layer_norm_115(c_void_p(buf434.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf461.data_ptr()))
    del arg81_1
    del arg82_1
    buf462 = reinterpret_tensor(buf440, (1568, 2048), (2048, 1), 0); del buf440  # reuse
    # Source Nodes: [x_300], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg260_1, reinterpret_tensor(buf461, (1568, 512), (512, 1), 0), reinterpret_tensor(arg259_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf462)
    del arg259_1
    del arg260_1
    buf463 = reinterpret_tensor(buf462, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf462  # reuse
    cpp_fused_gelu_116(c_void_p(buf463.data_ptr()))
    buf464 = reinterpret_tensor(buf461, (1568, 512), (512, 1), 0); del buf461  # reuse
    # Source Nodes: [x_304], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg262_1, reinterpret_tensor(buf463, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg261_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf464)
    del arg261_1
    del arg262_1
    buf465 = buf459; del buf459  # reuse
    buf466 = buf458; del buf458  # reuse
    buf468 = reinterpret_tensor(buf394, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf394  # reuse
    cpp_fused_add_native_layer_norm_117(c_void_p(buf434.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()))
    del arg83_1
    del arg84_1
    buf469 = buf446; del buf446  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg264_1, reinterpret_tensor(buf468, (1568, 512), (512, 1), 0), reinterpret_tensor(arg263_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf469)
    del arg263_1
    del arg264_1
    buf470 = reinterpret_tensor(buf468, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf468  # reuse
    buf471 = reinterpret_tensor(buf387, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf387  # reuse
    cpp_fused_clone_mul_118(c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    buf472 = reinterpret_tensor(buf453, (128, 196, 196), (38416, 196, 1), 0); del buf453  # reuse
    # Source Nodes: [x_308], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf470, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf471, (128, 32, 196), (6272, 196, 1), 0), out=buf472)
    buf473 = buf452; del buf452  # reuse
    buf474 = reinterpret_tensor(buf472, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf472  # reuse
    buf475 = buf450; del buf450  # reuse
    buf476 = reinterpret_tensor(buf474, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf474  # reuse
    buf477 = reinterpret_tensor(buf471, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf471  # reuse
    cpp_fused__softmax_clone_119(c_void_p(buf476.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()))
    buf478 = reinterpret_tensor(buf470, (128, 196, 32), (6272, 32, 1), 0); del buf470  # reuse
    # Source Nodes: [x_308], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf476, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf477, (128, 196, 32), (6272, 32, 1), 0), out=buf478)
    buf479 = reinterpret_tensor(buf477, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf477  # reuse
    cpp_fused_clone_120(c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()))
    buf480 = reinterpret_tensor(buf478, (1568, 512), (512, 1), 0); del buf478  # reuse
    # Source Nodes: [x_310], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg266_1, reinterpret_tensor(buf479, (1568, 512), (512, 1), 0), reinterpret_tensor(arg265_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf480)
    del arg265_1
    del arg266_1
    buf481 = reinterpret_tensor(buf480, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf480  # reuse
    buf482 = buf466; del buf466  # reuse
    buf483 = buf465; del buf465  # reuse
    buf485 = reinterpret_tensor(buf479, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf479  # reuse
    cpp_fused_add_native_layer_norm_121(c_void_p(buf481.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()))
    del arg85_1
    del arg86_1
    buf486 = reinterpret_tensor(buf463, (1568, 2048), (2048, 1), 0); del buf463  # reuse
    # Source Nodes: [x_314], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf485, (1568, 512), (512, 1), 0), reinterpret_tensor(arg267_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf486)
    del arg267_1
    del arg268_1
    buf487 = reinterpret_tensor(buf486, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf486  # reuse
    cpp_fused_gelu_122(c_void_p(buf487.data_ptr()))
    buf488 = reinterpret_tensor(buf485, (1568, 512), (512, 1), 0); del buf485  # reuse
    # Source Nodes: [x_318], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg270_1, reinterpret_tensor(buf487, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg269_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf488)
    del arg269_1
    del arg270_1
    buf489 = buf483; del buf483  # reuse
    buf490 = buf482; del buf482  # reuse
    buf492 = reinterpret_tensor(buf464, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf464  # reuse
    cpp_fused_add_native_layer_norm_123(c_void_p(buf481.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf492.data_ptr()))
    del arg87_1
    del arg88_1
    buf493 = buf469; del buf469  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg272_1, reinterpret_tensor(buf492, (1568, 512), (512, 1), 0), reinterpret_tensor(arg271_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf493)
    del arg271_1
    del arg272_1
    buf494 = reinterpret_tensor(buf492, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf492  # reuse
    buf495 = reinterpret_tensor(buf457, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf457  # reuse
    cpp_fused_clone_mul_124(c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    buf496 = reinterpret_tensor(buf476, (128, 196, 196), (38416, 196, 1), 0); del buf476  # reuse
    # Source Nodes: [x_322], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf494, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf495, (128, 32, 196), (6272, 196, 1), 0), out=buf496)
    buf497 = buf475; del buf475  # reuse
    buf498 = reinterpret_tensor(buf496, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf496  # reuse
    buf499 = buf473; del buf473  # reuse
    buf500 = reinterpret_tensor(buf498, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf498  # reuse
    buf501 = reinterpret_tensor(buf495, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf495  # reuse
    cpp_fused__softmax_clone_125(c_void_p(buf500.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()))
    buf502 = reinterpret_tensor(buf494, (128, 196, 32), (6272, 32, 1), 0); del buf494  # reuse
    # Source Nodes: [x_322], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf500, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf501, (128, 196, 32), (6272, 32, 1), 0), out=buf502)
    buf503 = reinterpret_tensor(buf501, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf501  # reuse
    cpp_fused_clone_126(c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()))
    buf504 = reinterpret_tensor(buf502, (1568, 512), (512, 1), 0); del buf502  # reuse
    # Source Nodes: [x_324], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg274_1, reinterpret_tensor(buf503, (1568, 512), (512, 1), 0), reinterpret_tensor(arg273_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf504)
    del arg273_1
    del arg274_1
    buf505 = buf490; del buf490  # reuse
    buf506 = buf489; del buf489  # reuse
    buf508 = reinterpret_tensor(buf503, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf503  # reuse
    cpp_fused_add_native_layer_norm_127(c_void_p(buf481.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf508.data_ptr()))
    del arg89_1
    del arg90_1
    buf509 = reinterpret_tensor(buf487, (1568, 2048), (2048, 1), 0); del buf487  # reuse
    # Source Nodes: [x_328], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg276_1, reinterpret_tensor(buf508, (1568, 512), (512, 1), 0), reinterpret_tensor(arg275_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf509)
    del arg275_1
    del arg276_1
    buf510 = reinterpret_tensor(buf509, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf509  # reuse
    cpp_fused_gelu_128(c_void_p(buf510.data_ptr()))
    buf511 = reinterpret_tensor(buf508, (1568, 512), (512, 1), 0); del buf508  # reuse
    # Source Nodes: [x_332], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf510, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg277_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf511)
    del arg277_1
    del arg278_1
    buf512 = buf506; del buf506  # reuse
    buf513 = buf505; del buf505  # reuse
    buf515 = reinterpret_tensor(buf441, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf441  # reuse
    cpp_fused_add_native_layer_norm_129(c_void_p(buf481.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf515.data_ptr()))
    del arg91_1
    del arg92_1
    buf516 = buf493; del buf493  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg280_1, reinterpret_tensor(buf515, (1568, 512), (512, 1), 0), reinterpret_tensor(arg279_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf516)
    del arg279_1
    del arg280_1
    buf517 = reinterpret_tensor(buf515, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf515  # reuse
    buf518 = reinterpret_tensor(buf434, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf434  # reuse
    cpp_fused_clone_mul_130(c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    buf519 = reinterpret_tensor(buf500, (128, 196, 196), (38416, 196, 1), 0); del buf500  # reuse
    # Source Nodes: [x_336], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf517, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf518, (128, 32, 196), (6272, 196, 1), 0), out=buf519)
    buf520 = buf499; del buf499  # reuse
    buf521 = reinterpret_tensor(buf519, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf519  # reuse
    buf522 = buf497; del buf497  # reuse
    buf523 = reinterpret_tensor(buf521, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf521  # reuse
    buf524 = reinterpret_tensor(buf518, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf518  # reuse
    cpp_fused__softmax_clone_131(c_void_p(buf523.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf524.data_ptr()))
    buf525 = reinterpret_tensor(buf517, (128, 196, 32), (6272, 32, 1), 0); del buf517  # reuse
    # Source Nodes: [x_336], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf523, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf524, (128, 196, 32), (6272, 32, 1), 0), out=buf525)
    buf526 = reinterpret_tensor(buf524, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf524  # reuse
    cpp_fused_clone_132(c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()))
    buf527 = reinterpret_tensor(buf525, (1568, 512), (512, 1), 0); del buf525  # reuse
    # Source Nodes: [x_338], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg282_1, reinterpret_tensor(buf526, (1568, 512), (512, 1), 0), reinterpret_tensor(arg281_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf527)
    del arg281_1
    del arg282_1
    buf528 = reinterpret_tensor(buf527, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf527  # reuse
    buf529 = buf513; del buf513  # reuse
    buf530 = buf512; del buf512  # reuse
    buf532 = reinterpret_tensor(buf526, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf526  # reuse
    cpp_fused_add_native_layer_norm_133(c_void_p(buf528.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf532.data_ptr()))
    del arg93_1
    del arg94_1
    buf533 = reinterpret_tensor(buf510, (1568, 2048), (2048, 1), 0); del buf510  # reuse
    # Source Nodes: [x_342], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg284_1, reinterpret_tensor(buf532, (1568, 512), (512, 1), 0), reinterpret_tensor(arg283_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf533)
    del arg283_1
    del arg284_1
    buf534 = reinterpret_tensor(buf533, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf533  # reuse
    cpp_fused_gelu_134(c_void_p(buf534.data_ptr()))
    buf535 = reinterpret_tensor(buf532, (1568, 512), (512, 1), 0); del buf532  # reuse
    # Source Nodes: [x_346], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg286_1, reinterpret_tensor(buf534, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg285_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf535)
    del arg285_1
    del arg286_1
    buf536 = buf530; del buf530  # reuse
    buf537 = buf529; del buf529  # reuse
    buf539 = reinterpret_tensor(buf511, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf511  # reuse
    cpp_fused_add_native_layer_norm_135(c_void_p(buf528.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf539.data_ptr()))
    del arg95_1
    del arg96_1
    buf540 = buf516; del buf516  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg288_1, reinterpret_tensor(buf539, (1568, 512), (512, 1), 0), reinterpret_tensor(arg287_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf540)
    del arg287_1
    del arg288_1
    buf541 = reinterpret_tensor(buf539, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf539  # reuse
    buf542 = reinterpret_tensor(buf504, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf504  # reuse
    cpp_fused_clone_mul_136(c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()))
    buf543 = reinterpret_tensor(buf523, (128, 196, 196), (38416, 196, 1), 0); del buf523  # reuse
    # Source Nodes: [x_350], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf541, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf542, (128, 32, 196), (6272, 196, 1), 0), out=buf543)
    buf544 = buf522; del buf522  # reuse
    buf545 = reinterpret_tensor(buf543, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf543  # reuse
    buf546 = buf520; del buf520  # reuse
    buf547 = reinterpret_tensor(buf545, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf545  # reuse
    buf548 = reinterpret_tensor(buf542, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf542  # reuse
    cpp_fused__softmax_clone_137(c_void_p(buf547.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf548.data_ptr()))
    buf549 = reinterpret_tensor(buf541, (128, 196, 32), (6272, 32, 1), 0); del buf541  # reuse
    # Source Nodes: [x_350], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf547, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf548, (128, 196, 32), (6272, 32, 1), 0), out=buf549)
    buf550 = reinterpret_tensor(buf548, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf548  # reuse
    cpp_fused_clone_138(c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()))
    buf551 = reinterpret_tensor(buf549, (1568, 512), (512, 1), 0); del buf549  # reuse
    # Source Nodes: [x_352], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg290_1, reinterpret_tensor(buf550, (1568, 512), (512, 1), 0), reinterpret_tensor(arg289_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf551)
    del arg289_1
    del arg290_1
    buf552 = buf537; del buf537  # reuse
    buf553 = buf536; del buf536  # reuse
    buf555 = reinterpret_tensor(buf550, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf550  # reuse
    cpp_fused_add_native_layer_norm_139(c_void_p(buf528.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf555.data_ptr()))
    del arg97_1
    del arg98_1
    buf556 = reinterpret_tensor(buf534, (1568, 2048), (2048, 1), 0); del buf534  # reuse
    # Source Nodes: [x_356], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg292_1, reinterpret_tensor(buf555, (1568, 512), (512, 1), 0), reinterpret_tensor(arg291_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf556)
    del arg291_1
    del arg292_1
    buf557 = reinterpret_tensor(buf556, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf556  # reuse
    cpp_fused_gelu_140(c_void_p(buf557.data_ptr()))
    buf558 = reinterpret_tensor(buf555, (1568, 512), (512, 1), 0); del buf555  # reuse
    # Source Nodes: [x_360], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg294_1, reinterpret_tensor(buf557, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg293_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf558)
    del arg293_1
    del arg294_1
    buf559 = buf553; del buf553  # reuse
    buf560 = buf552; del buf552  # reuse
    buf562 = reinterpret_tensor(buf488, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf488  # reuse
    cpp_fused_add_native_layer_norm_141(c_void_p(buf528.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf562.data_ptr()))
    del arg100_1
    del arg99_1
    buf563 = buf540; del buf540  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg296_1, reinterpret_tensor(buf562, (1568, 512), (512, 1), 0), reinterpret_tensor(arg295_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf563)
    del arg295_1
    del arg296_1
    buf564 = reinterpret_tensor(buf562, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf562  # reuse
    buf565 = reinterpret_tensor(buf481, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf481  # reuse
    cpp_fused_clone_mul_142(c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()))
    buf566 = reinterpret_tensor(buf547, (128, 196, 196), (38416, 196, 1), 0); del buf547  # reuse
    # Source Nodes: [x_364], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf564, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf565, (128, 32, 196), (6272, 196, 1), 0), out=buf566)
    buf567 = buf546; del buf546  # reuse
    buf568 = reinterpret_tensor(buf566, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf566  # reuse
    buf569 = buf544; del buf544  # reuse
    buf570 = reinterpret_tensor(buf568, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf568  # reuse
    buf571 = reinterpret_tensor(buf565, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf565  # reuse
    cpp_fused__softmax_clone_143(c_void_p(buf570.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf567.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf571.data_ptr()))
    del buf563
    del buf567
    del buf569
    buf572 = reinterpret_tensor(buf564, (128, 196, 32), (6272, 32, 1), 0); del buf564  # reuse
    # Source Nodes: [x_364], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf570, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf571, (128, 196, 32), (6272, 32, 1), 0), out=buf572)
    del buf570
    buf573 = reinterpret_tensor(buf571, (8, 1, 196, 32, 16), (100352, 100352, 512, 16, 1), 0); del buf571  # reuse
    cpp_fused_clone_144(c_void_p(buf572.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = reinterpret_tensor(buf572, (1568, 512), (512, 1), 0); del buf572  # reuse
    # Source Nodes: [x_366], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg298_1, reinterpret_tensor(buf573, (1568, 512), (512, 1), 0), reinterpret_tensor(arg297_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf574)
    del arg297_1
    del arg298_1
    buf575 = reinterpret_tensor(buf574, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf574  # reuse
    buf576 = buf560; del buf560  # reuse
    buf577 = buf559; del buf559  # reuse
    buf579 = reinterpret_tensor(buf573, (8, 1, 196, 512), (100352, 100352, 512, 1), 0); del buf573  # reuse
    cpp_fused_add_native_layer_norm_145(c_void_p(buf575.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf579.data_ptr()))
    del arg101_1
    del arg102_1
    del buf528
    del buf535
    del buf551
    del buf558
    buf580 = reinterpret_tensor(buf557, (1568, 2048), (2048, 1), 0); del buf557  # reuse
    # Source Nodes: [x_370], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg300_1, reinterpret_tensor(buf579, (1568, 512), (512, 1), 0), reinterpret_tensor(arg299_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf580)
    del arg299_1
    del arg300_1
    buf581 = reinterpret_tensor(buf580, (8, 1, 196, 2048), (401408, 401408, 2048, 1), 0); del buf580  # reuse
    cpp_fused_gelu_146(c_void_p(buf581.data_ptr()))
    buf582 = reinterpret_tensor(buf579, (1568, 512), (512, 1), 0); del buf579  # reuse
    # Source Nodes: [x_374], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg302_1, reinterpret_tensor(buf581, (1568, 2048), (2048, 1), 0), reinterpret_tensor(arg301_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf582)
    del arg301_1
    del arg302_1
    del buf581
    buf583 = reinterpret_tensor(buf577, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf577  # reuse
    buf584 = reinterpret_tensor(buf576, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf576  # reuse
    buf586 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf587 = reinterpret_tensor(buf586, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf586  # reuse
    cpp_fused_mean_native_layer_norm_147(c_void_p(buf587.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()))
    del arg103_1
    del arg104_1
    del buf575
    del buf582
    del buf583
    del buf584
    buf588 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_389], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg304_1, reinterpret_tensor(buf587, (8, 512), (512, 1), 0), reinterpret_tensor(arg303_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf588)
    del arg303_1
    del arg304_1
    return (buf588, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 16, 196, 128), (401408, 25088, 128, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((1, 4, 196, 256), (200704, 50176, 256, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((1, 1, 196, 512), (100352, 100352, 512, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((1000, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('jx_nest_base', benchmark_compiled_module)
