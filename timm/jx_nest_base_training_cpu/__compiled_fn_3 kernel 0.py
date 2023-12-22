
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (16L*x1) + (48L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (48L*x0))] = tmp0;
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
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
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
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
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr3[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (2048L*x2) + (401408L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((16L*x1) + (16L*x1_inner) + (2048L*(static_cast<long>(x0) % static_cast<long>(196L))) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(3136L))) + (100352L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (c10::div_floor_integer((x1 + x1_inner), 4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (2048L*x2) + (401408L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((16L*x1) + (16L*x1_inner) + (2048L*(static_cast<long>(x0) % static_cast<long>(196L))) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_7 = async_compile.cpp('''
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
                       float* out_ptr3)
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (2048L*x2) + (401408L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((16L*x1) + (16L*x1_inner) + (2048L*(static_cast<long>(x0) % static_cast<long>(196L))) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(3136L))) + (100352L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (c10::div_floor_integer((x1 + x1_inner), 4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_12 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp8 = in_ptr4[static_cast<long>(x0)];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            auto tmp9 = static_cast<float>(0.9782608691602945);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp7 * tmp11;
                            auto tmp13 = tmp6 + tmp12;
                            tmp13.store(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (128L*x2_inner) + (25088L*x1) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(128.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-06);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (16L*x3) + (2048L*x2) + (2048L*x2_inner) + (401408L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (25088L*x1) + (401408L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(128.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-06);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (16L*x3) + (2048L*x2) + (401408L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((16L*x1) + (16L*x1_inner) + (2048L*(static_cast<long>(x0) % static_cast<long>(196L))) + (401408L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12845056L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_permute_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(14L))) + (25088L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 14L))) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (128L*(static_cast<long>(x2) % static_cast<long>(14L))) + (1792L*(static_cast<long>(x1) % static_cast<long>(14L))) + (25088L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 14L))) + (401408L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.9782608691602945);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_view_16 = async_compile.cpp('''
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
                       long* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
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
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (802816L*x0)));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (56L*x2) + (3136L*x0))] = static_cast<float>(tmp_acc0.mean);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (56L*x1) + (56L*x1_inner) + (3136L*x0)));
                                auto tmp1 = static_cast<float>(256.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (56L*x2) + (3136L*x0)), static_cast<long>(56L));
                        }
                    }
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
                                auto tmp8 = out_ptr0[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = out_ptr2[static_cast<long>(x1 + (56L*x2) + (3136L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp10 * tmp12;
                                auto tmp14 = masked_load(in_ptr1 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp15 = tmp13 * tmp14;
                                auto tmp16 = masked_load(in_ptr2 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp17 = tmp15 + tmp16;
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp6(), to_float_mask(tmp5));
                            tmp18.store(out_ptr3 + static_cast<long>(x3 + (256L*x2) + (14592L*x1) + (831744L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp1 = out_ptr3[static_cast<long>(256L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp3 = out_ptr3[static_cast<long>(512L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp5 = out_ptr3[static_cast<long>(14592L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp7 = out_ptr3[static_cast<long>(14848L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp9 = out_ptr3[static_cast<long>(15104L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp11 = out_ptr3[static_cast<long>(29184L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp13 = out_ptr3[static_cast<long>(29440L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp15 = out_ptr3[static_cast<long>(29696L + x1 + (512L*x3) + (29184L*x2) + (831744L*x0))];
                            auto tmp2 = max_propagate_nan(tmp1, tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp10 = max_propagate_nan(tmp9, tmp8);
                            auto tmp12 = max_propagate_nan(tmp11, tmp10);
                            auto tmp14 = max_propagate_nan(tmp13, tmp12);
                            auto tmp16 = max_propagate_nan(tmp15, tmp14);
                            auto tmp17 = tmp1 > tmp0;
                            auto tmp18 = c10::convert<long>(1L + (2L*x3) + (114L*x2));
                            auto tmp19 = c10::convert<long>((2L*x3) + (114L*x2));
                            auto tmp20 = tmp17 ? tmp18 : tmp19;
                            auto tmp21 = tmp3 > tmp2;
                            auto tmp22 = c10::convert<long>(2L + (2L*x3) + (114L*x2));
                            auto tmp23 = tmp21 ? tmp22 : tmp20;
                            auto tmp24 = tmp5 > tmp4;
                            auto tmp25 = c10::convert<long>(57L + (2L*x3) + (114L*x2));
                            auto tmp26 = tmp24 ? tmp25 : tmp23;
                            auto tmp27 = tmp7 > tmp6;
                            auto tmp28 = c10::convert<long>(58L + (2L*x3) + (114L*x2));
                            auto tmp29 = tmp27 ? tmp28 : tmp26;
                            auto tmp30 = tmp9 > tmp8;
                            auto tmp31 = c10::convert<long>(59L + (2L*x3) + (114L*x2));
                            auto tmp32 = tmp30 ? tmp31 : tmp29;
                            auto tmp33 = tmp11 > tmp10;
                            auto tmp34 = c10::convert<long>(114L + (2L*x3) + (114L*x2));
                            auto tmp35 = tmp33 ? tmp34 : tmp32;
                            auto tmp36 = tmp13 > tmp12;
                            auto tmp37 = c10::convert<long>(115L + (2L*x3) + (114L*x2));
                            auto tmp38 = tmp36 ? tmp37 : tmp35;
                            auto tmp39 = tmp15 > tmp14;
                            auto tmp40 = c10::convert<long>(116L + (2L*x3) + (114L*x2));
                            auto tmp41 = tmp39 ? tmp40 : tmp38;
                            out_ptr4[static_cast<long>(x1 + (256L*x3) + (7168L*x2) + (200704L*x0))] = tmp16;
                            out_ptr5[static_cast<long>(x1 + (256L*x3) + (7168L*x2) + (200704L*x0))] = tmp41;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(2L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 2L))) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)));
                                auto tmp2 = tmp0 + tmp1;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr6[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr7[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(2L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 2L))) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)));
                            auto tmp3 = out_ptr6[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp6 = out_ptr7[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr8[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (1024L*x2) + (200704L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr8[static_cast<long>((4L*x1) + (4L*x1_inner) + (1024L*(static_cast<long>(x0) % static_cast<long>(196L))) + (200704L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr9 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_17 = async_compile.cpp('''
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


cpp_fused__softmax_clone_18 = async_compile.cpp('''
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


cpp_fused_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(784L))) + (25088L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(8L))) + (200704L*(c10::div_floor_integer(x0, 784L))) + (c10::div_floor_integer((x1 + x1_inner), 8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_21 = async_compile.cpp('''
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
                       float* out_ptr3)
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
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(2L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 2L))) + (200704L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                                auto tmp4 = in_ptr3[static_cast<long>(x0)];
                                auto tmp2 = tmp0 + tmp1;
                                auto tmp5 = static_cast<float>(0.9565217383205891);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                                auto tmp8 = tmp3 * tmp7;
                                auto tmp9 = tmp2 + tmp8;
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp9);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0.mean);
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(2L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 2L))) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp4 = in_ptr3[static_cast<long>(x0)];
                            auto tmp10 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp13 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp5 = static_cast<float>(0.9565217383205891);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp3 * tmp7;
                            auto tmp9 = tmp2 + tmp8;
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp9 - tmp11;
                            auto tmp14 = static_cast<float>(256.0);
                            auto tmp15 = tmp13 / tmp14;
                            auto tmp16 = static_cast<float>(1e-06);
                            auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                            auto tmp18 = 1 / std::sqrt(tmp17);
                            auto tmp19 = at::vec::Vectorized<float>(tmp18);
                            auto tmp20 = tmp12 * tmp19;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp20.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (1024L*x2) + (200704L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((4L*x1) + (4L*x1_inner) + (1024L*(static_cast<long>(x0) % static_cast<long>(196L))) + (200704L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_24 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(2L))) + (7168L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 2L))) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp4 = in_ptr3[static_cast<long>(x0)];
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp11 = in_ptr4[static_cast<long>(x0)];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp5 = static_cast<float>(0.9565217383205891);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp3 * tmp7;
                            auto tmp9 = tmp2 + tmp8;
                            auto tmp12 = tmp11 / tmp5;
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp10 * tmp13;
                            auto tmp15 = tmp9 + tmp14;
                            tmp15.store(in_out_ptr0 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (256L*x2) + (256L*x2_inner) + (50176L*x1) + (200704L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (784L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(256.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-06);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (4L*x3) + (1024L*x2) + (1024L*x2_inner) + (200704L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(256.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-06);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (4L*x3) + (1024L*x2) + (200704L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((4L*x1) + (4L*x1_inner) + (1024L*(static_cast<long>(x0) % static_cast<long>(196L))) + (200704L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_25 = async_compile.cpp('''
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


cpp_fused__softmax_clone_26 = async_compile.cpp('''
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


cpp_fused_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(784L))) + (25088L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(8L))) + (200704L*(c10::div_floor_integer(x0, 784L))) + (c10::div_floor_integer((x1 + x1_inner), 8L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.9347826093435287);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*x2) + (50176L*x1) + (200704L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp8 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp3 = static_cast<float>(0.9347826093435287);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 - tmp9;
                            auto tmp12 = static_cast<float>(256.0);
                            auto tmp13 = tmp11 / tmp12;
                            auto tmp14 = static_cast<float>(1e-06);
                            auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                            auto tmp16 = 1 / std::sqrt(tmp15);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp10 * tmp17;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp18.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (4L*x3) + (4L*x3_inner) + (1024L*x2) + (200704L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((4L*x1) + (4L*x1_inner) + (1024L*(static_cast<long>(x0) % static_cast<long>(196L))) + (200704L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 196L)) % static_cast<long>(4L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_permute_32 = async_compile.cpp('''
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(14L))) + (50176L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 14L))) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(14L))) + (50176L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 14L))) + (200704L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (256L*(static_cast<long>(x2) % static_cast<long>(14L))) + (3584L*(static_cast<long>(x1) % static_cast<long>(14L))) + (50176L*(c10::div_floor_integer(x2, 14L))) + (100352L*(c10::div_floor_integer(x1, 14L))) + (200704L*x0)));
                            auto tmp9 = in_ptr4[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.9347826093435287);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            auto tmp10 = tmp9 / tmp3;
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp8 * tmp11;
                            auto tmp13 = tmp7 + tmp12;
                            tmp13.store(out_ptr0 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_view_33 = async_compile.cpp('''
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
                       long* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9)
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
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (401408L*x0)));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = static_cast<float>(tmp_acc0.mean);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp5;
                        }
                    }
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
                                auto tmp8 = out_ptr0[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp7 - tmp9;
                                auto tmp11 = out_ptr2[static_cast<long>(x1 + (28L*x2) + (784L*x0))];
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp10 * tmp12;
                                auto tmp14 = masked_load(in_ptr1 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp15 = tmp13 * tmp14;
                                auto tmp16 = masked_load(in_ptr2 + static_cast<long>(x3), to_float_mask(tmp5));
                                auto tmp17 = tmp15 + tmp16;
                                return tmp17;
                            }
                            ;
                            auto tmp18 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity()), tmp6(), to_float_mask(tmp5));
                            tmp18.store(out_ptr3 + static_cast<long>(x3 + (512L*x2) + (14848L*x1) + (430592L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr3[static_cast<long>(x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp1 = out_ptr3[static_cast<long>(512L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp3 = out_ptr3[static_cast<long>(1024L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp5 = out_ptr3[static_cast<long>(14848L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp7 = out_ptr3[static_cast<long>(15360L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp9 = out_ptr3[static_cast<long>(15872L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp11 = out_ptr3[static_cast<long>(29696L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp13 = out_ptr3[static_cast<long>(30208L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp15 = out_ptr3[static_cast<long>(30720L + x1 + (1024L*x3) + (29696L*x2) + (430592L*x0))];
                            auto tmp2 = max_propagate_nan(tmp1, tmp0);
                            auto tmp4 = max_propagate_nan(tmp3, tmp2);
                            auto tmp6 = max_propagate_nan(tmp5, tmp4);
                            auto tmp8 = max_propagate_nan(tmp7, tmp6);
                            auto tmp10 = max_propagate_nan(tmp9, tmp8);
                            auto tmp12 = max_propagate_nan(tmp11, tmp10);
                            auto tmp14 = max_propagate_nan(tmp13, tmp12);
                            auto tmp16 = max_propagate_nan(tmp15, tmp14);
                            auto tmp17 = tmp1 > tmp0;
                            auto tmp18 = c10::convert<long>(1L + (2L*x3) + (58L*x2));
                            auto tmp19 = c10::convert<long>((2L*x3) + (58L*x2));
                            auto tmp20 = tmp17 ? tmp18 : tmp19;
                            auto tmp21 = tmp3 > tmp2;
                            auto tmp22 = c10::convert<long>(2L + (2L*x3) + (58L*x2));
                            auto tmp23 = tmp21 ? tmp22 : tmp20;
                            auto tmp24 = tmp5 > tmp4;
                            auto tmp25 = c10::convert<long>(29L + (2L*x3) + (58L*x2));
                            auto tmp26 = tmp24 ? tmp25 : tmp23;
                            auto tmp27 = tmp7 > tmp6;
                            auto tmp28 = c10::convert<long>(30L + (2L*x3) + (58L*x2));
                            auto tmp29 = tmp27 ? tmp28 : tmp26;
                            auto tmp30 = tmp9 > tmp8;
                            auto tmp31 = c10::convert<long>(31L + (2L*x3) + (58L*x2));
                            auto tmp32 = tmp30 ? tmp31 : tmp29;
                            auto tmp33 = tmp11 > tmp10;
                            auto tmp34 = c10::convert<long>(58L + (2L*x3) + (58L*x2));
                            auto tmp35 = tmp33 ? tmp34 : tmp32;
                            auto tmp36 = tmp13 > tmp12;
                            auto tmp37 = c10::convert<long>(59L + (2L*x3) + (58L*x2));
                            auto tmp38 = tmp36 ? tmp37 : tmp35;
                            auto tmp39 = tmp15 > tmp14;
                            auto tmp40 = c10::convert<long>(60L + (2L*x3) + (58L*x2));
                            auto tmp41 = tmp39 ? tmp40 : tmp38;
                            out_ptr4[static_cast<long>(x1 + (512L*x3) + (7168L*x2) + (100352L*x0))] = tmp16;
                            out_ptr5[static_cast<long>(x1 + (512L*x3) + (7168L*x2) + (100352L*x0))] = tmp41;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr6[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr7[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1)));
                        auto tmp3 = out_ptr6[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr7[static_cast<long>(x1 + (196L*x0))];
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
                        tmp13.store(out_ptr8 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr8 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr9 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_38 = async_compile.cpp('''
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
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp4 = in_ptr3[static_cast<long>(x0)];
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp5 = static_cast<float>(0.9130434766411781);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = at::vec::Vectorized<float>(tmp6);
                            auto tmp8 = tmp3 * tmp7;
                            auto tmp9 = tmp2 + tmp8;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp9);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp4 = in_ptr3[static_cast<long>(x0)];
                        auto tmp10 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp13 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = static_cast<float>(0.9130434766411781);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp3 * tmp7;
                        auto tmp9 = tmp2 + tmp8;
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 - tmp11;
                        auto tmp14 = static_cast<float>(512.0);
                        auto tmp15 = tmp13 / tmp14;
                        auto tmp16 = static_cast<float>(1e-06);
                        auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                        auto tmp18 = 1 / std::sqrt(tmp17);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_41 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp4 = in_ptr3[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp11 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = static_cast<float>(0.9130434766411781);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 * tmp7;
                    auto tmp9 = tmp2 + tmp8;
                    auto tmp12 = tmp11 / tmp5;
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp10 * tmp13;
                    auto tmp15 = tmp9 + tmp14;
                    tmp15.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_42 = async_compile.cpp('''
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


cpp_fused__softmax_clone_43 = async_compile.cpp('''
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


cpp_fused_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.8913043439388275);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.8913043439388275);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_49 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.8913043439388275);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_50 = async_compile.cpp('''
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


cpp_fused__softmax_clone_51 = async_compile.cpp('''
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


cpp_fused_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.8695652186870575);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.8695652186870575);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_57 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.8695652186870575);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.8478260785341263);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.8478260785341263);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_65 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.8478260785341263);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_66 = async_compile.cpp('''
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


cpp_fused__softmax_clone_67 = async_compile.cpp('''
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


cpp_fused_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.8260869532823563);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.8260869532823563);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_73 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.8260869532823563);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_74 = async_compile.cpp('''
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


cpp_fused__softmax_clone_75 = async_compile.cpp('''
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


cpp_fused_view_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.8043478280305862);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.8043478280305862);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_81 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.8043478280305862);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_view_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.782608687877655);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.782608687877655);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_89 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.782608687877655);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_90 = async_compile.cpp('''
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


cpp_fused__softmax_clone_91 = async_compile.cpp('''
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


cpp_fused_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.760869562625885);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.760869562625885);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_97 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.760869562625885);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_98 = async_compile.cpp('''
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


cpp_fused__softmax_clone_99 = async_compile.cpp('''
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


cpp_fused_view_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.739130437374115);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.739130437374115);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_105 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.739130437374115);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_view_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.717391312122345);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.717391312122345);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_113 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.717391312122345);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_114 = async_compile.cpp('''
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


cpp_fused__softmax_clone_115 = async_compile.cpp('''
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


cpp_fused_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.695652186870575);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.695652186870575);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_120 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_121 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.695652186870575);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_122 = async_compile.cpp('''
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


cpp_fused__softmax_clone_123 = async_compile.cpp('''
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


cpp_fused_view_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.6739130616188049);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.6739130616188049);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_129 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.6739130616188049);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_view_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.6521739065647125);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.6521739065647125);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_135 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_137 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.6521739065647125);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_138 = async_compile.cpp('''
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


cpp_fused__softmax_clone_139 = async_compile.cpp('''
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


cpp_fused_view_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_141 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_142 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.6304347813129425);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.6304347813129425);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_145 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.6304347813129425);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_146 = async_compile.cpp('''
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


cpp_fused__softmax_clone_147 = async_compile.cpp('''
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


cpp_fused_view_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_150 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.6086956560611725);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.6086956560611725);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_153 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.6086956560611725);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_154 = async_compile.cpp('''
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


cpp_fused__softmax_clone_155 = async_compile.cpp('''
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


cpp_fused_view_156 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_157 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.5869565308094025);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.5869565308094025);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_160 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_161 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.5869565308094025);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_162 = async_compile.cpp('''
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


cpp_fused__softmax_clone_163 = async_compile.cpp('''
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


cpp_fused_view_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_165 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.5652174055576324);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.5652174055576324);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_169 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.5652174055576324);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_170 = async_compile.cpp('''
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


cpp_fused__softmax_clone_171 = async_compile.cpp('''
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


cpp_fused_view_172 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.54347825050354);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.54347825050354);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_175 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_176 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_177 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.54347825050354);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_178 = async_compile.cpp('''
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


cpp_fused__softmax_clone_179 = async_compile.cpp('''
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


cpp_fused_view_180 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_181 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_182 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.52173912525177);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.52173912525177);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_183 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_184 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_185 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.52173912525177);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_186 = async_compile.cpp('''
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


cpp_fused__softmax_clone_187 = async_compile.cpp('''
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


cpp_fused_view_188 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(16L))) + (100352L*(c10::div_floor_integer(x0, 196L))) + (c10::div_floor_integer((x1 + x1_inner), 16L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_bernoulli_189 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_view_190 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                            auto tmp2 = in_ptr2[static_cast<long>(x0)];
                            auto tmp3 = static_cast<float>(0.5);
                            auto tmp4 = tmp2 / tmp3;
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp1 * tmp5;
                            auto tmp7 = tmp0 + tmp6;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp7);
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                        auto tmp2 = in_ptr2[static_cast<long>(x0)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp11 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = static_cast<float>(0.5);
                        auto tmp4 = tmp2 / tmp3;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp1 * tmp5;
                        auto tmp7 = tmp0 + tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp12 = static_cast<float>(512.0);
                        auto tmp13 = tmp11 / tmp12;
                        auto tmp14 = static_cast<float>(1e-06);
                        auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                        auto tmp16 = 1 / std::sqrt(tmp15);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp10 * tmp17;
                        tmp18.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_191 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
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
                tmp11.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_bernoulli_192 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = std::numeric_limits<float>::quiet_NaN();
            auto tmp1 = at::vec::Vectorized<float>(tmp0);
            tmp1.store(out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_mean_mul_native_layer_norm_193 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto out_ptr3 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp2 = in_ptr2[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
                    auto tmp9 = in_ptr3[static_cast<long>(x0)];
                    auto tmp3 = static_cast<float>(0.5);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 * tmp5;
                    auto tmp7 = tmp0 + tmp6;
                    auto tmp10 = tmp9 / tmp3;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp8 * tmp11;
                    auto tmp13 = tmp7 + tmp12;
                    tmp13.store(in_out_ptr0 + static_cast<long>(x1 + (100352L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (512L*x2_inner) + (7168L*x1) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(512.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-06);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (7168L*x2) + (7168L*x2_inner) + (100352L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(512.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-06);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (7168L*x2) + (100352L*x0))] = tmp9;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (14L*x1) + (7168L*x2) + (100352L*x0)));
                                auto tmp1 = in_ptr4[static_cast<long>(x1)];
                                auto tmp4 = in_ptr5[static_cast<long>(x1)];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 * tmp2;
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                tmp_acc0_vec = tmp_acc0_vec + tmp6;
                            }
                            #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                            for(long x3=static_cast<long>(8L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = out_ptr2[static_cast<long>(x3 + (14L*x1) + (7168L*x2) + (100352L*x0))];
                                auto tmp1 = in_ptr4[static_cast<long>(x1)];
                                auto tmp3 = in_ptr5[static_cast<long>(x1)];
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                                tmp_acc0 = tmp_acc0 + tmp4;
                            }
                        }
                        tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                        out_ptr3[static_cast<long>(x1 + (512L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4096L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_194 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
                       float* in_out_ptr6,
                       float* in_out_ptr7,
                       float* in_out_ptr8,
                       float* in_out_ptr9,
                       float* in_out_ptr10,
                       float* in_out_ptr11,
                       float* in_out_ptr12,
                       float* in_out_ptr13,
                       float* in_out_ptr14,
                       float* in_out_ptr15,
                       float* in_out_ptr16,
                       float* in_out_ptr17,
                       float* in_out_ptr18,
                       float* in_out_ptr19,
                       float* in_out_ptr20,
                       float* in_out_ptr21,
                       float* in_out_ptr22,
                       float* in_out_ptr23,
                       float* in_out_ptr24,
                       float* in_out_ptr25,
                       float* in_out_ptr26,
                       float* in_out_ptr27,
                       float* in_out_ptr28,
                       float* in_out_ptr29,
                       float* in_out_ptr30,
                       float* in_out_ptr31,
                       float* in_out_ptr32,
                       float* in_out_ptr33,
                       float* in_out_ptr34,
                       float* in_out_ptr35,
                       float* in_out_ptr36,
                       float* in_out_ptr37,
                       float* in_out_ptr38,
                       float* in_out_ptr39,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                        auto tmp1 = static_cast<float>(512.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp4 = static_cast<float>(1e-06);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.rsqrt();
                        auto tmp8 = tmp7 / tmp2;
                        tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr0 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = tmp5 / tmp1;
                    out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr3 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr4 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr6 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr7 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr8 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr9 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr10 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr11 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr12 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr13 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr14 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr15 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr16 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr17 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr18 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr19 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr20 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr21 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr22 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr23 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr24 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr25 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr26 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr27 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr28 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr29 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr30 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr31 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr32 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr33 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr34 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr35 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr36 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr37 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr38 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-06);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr39 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = tmp5 / tmp1;
                    out_ptr1[static_cast<long>(x1 + (4L*x2) + (784L*x0))] = tmp6;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = tmp5 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (4L*x2) + (784L*x0))] = tmp6;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = tmp5 / tmp1;
                    out_ptr3[static_cast<long>(x1 + (4L*x2) + (784L*x0))] = tmp6;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = tmp5 / tmp1;
                    out_ptr4[static_cast<long>(x1 + (4L*x2) + (784L*x0))] = tmp6;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                        auto tmp1 = static_cast<float>(128.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp4 = static_cast<float>(1e-06);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.rsqrt();
                        auto tmp8 = tmp7 / tmp2;
                        tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr5 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr5 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                        auto tmp1 = static_cast<float>(128.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp4 = static_cast<float>(1e-06);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.rsqrt();
                        auto tmp8 = tmp7 / tmp2;
                        tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr6 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr6 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                        auto tmp1 = static_cast<float>(128.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp4 = static_cast<float>(1e-06);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.rsqrt();
                        auto tmp8 = tmp7 / tmp2;
                        tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                {
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                        auto tmp1 = static_cast<float>(128.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp4 = static_cast<float>(1e-06);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.rsqrt();
                        auto tmp8 = tmp7 / tmp2;
                        tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr8 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr8 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306 = args
    args.clear()
    assert_size_stride(primals_1, (1, 16, 196, 128), (401408, 25088, 128, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (1, 4, 196, 256), (200704, 50176, 256, 1))
    assert_size_stride(primals_13, (256, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (1, 1, 196, 512), (100352, 100352, 512, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, ), (1, ))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, ), (1, ))
    assert_size_stride(primals_35, (512, ), (1, ))
    assert_size_stride(primals_36, (512, ), (1, ))
    assert_size_stride(primals_37, (512, ), (1, ))
    assert_size_stride(primals_38, (512, ), (1, ))
    assert_size_stride(primals_39, (512, ), (1, ))
    assert_size_stride(primals_40, (512, ), (1, ))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (512, ), (1, ))
    assert_size_stride(primals_48, (512, ), (1, ))
    assert_size_stride(primals_49, (512, ), (1, ))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (512, ), (1, ))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (512, ), (1, ))
    assert_size_stride(primals_60, (512, ), (1, ))
    assert_size_stride(primals_61, (512, ), (1, ))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, ), (1, ))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, ), (1, ))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (512, ), (1, ))
    assert_size_stride(primals_80, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (512, ), (1, ))
    assert_size_stride(primals_102, (512, ), (1, ))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (384, 128), (128, 1))
    assert_size_stride(primals_109, (384, ), (1, ))
    assert_size_stride(primals_110, (128, 128), (128, 1))
    assert_size_stride(primals_111, (128, ), (1, ))
    assert_size_stride(primals_112, (512, 128), (128, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (128, 512), (512, 1))
    assert_size_stride(primals_115, (128, ), (1, ))
    assert_size_stride(primals_116, (384, 128), (128, 1))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (128, 128), (128, 1))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (512, 128), (128, 1))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (128, 512), (512, 1))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (768, 256), (256, 1))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (256, 256), (256, 1))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (1024, 256), (256, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (256, 1024), (1024, 1))
    assert_size_stride(primals_133, (256, ), (1, ))
    assert_size_stride(primals_134, (768, 256), (256, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (256, 256), (256, 1))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (1024, 256), (256, 1))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (256, 1024), (1024, 1))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (1536, 512), (512, 1))
    assert_size_stride(primals_145, (1536, ), (1, ))
    assert_size_stride(primals_146, (512, 512), (512, 1))
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (2048, 512), (512, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (512, 2048), (2048, 1))
    assert_size_stride(primals_151, (512, ), (1, ))
    assert_size_stride(primals_152, (1536, 512), (512, 1))
    assert_size_stride(primals_153, (1536, ), (1, ))
    assert_size_stride(primals_154, (512, 512), (512, 1))
    assert_size_stride(primals_155, (512, ), (1, ))
    assert_size_stride(primals_156, (2048, 512), (512, 1))
    assert_size_stride(primals_157, (2048, ), (1, ))
    assert_size_stride(primals_158, (512, 2048), (2048, 1))
    assert_size_stride(primals_159, (512, ), (1, ))
    assert_size_stride(primals_160, (1536, 512), (512, 1))
    assert_size_stride(primals_161, (1536, ), (1, ))
    assert_size_stride(primals_162, (512, 512), (512, 1))
    assert_size_stride(primals_163, (512, ), (1, ))
    assert_size_stride(primals_164, (2048, 512), (512, 1))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_166, (512, 2048), (2048, 1))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_168, (1536, 512), (512, 1))
    assert_size_stride(primals_169, (1536, ), (1, ))
    assert_size_stride(primals_170, (512, 512), (512, 1))
    assert_size_stride(primals_171, (512, ), (1, ))
    assert_size_stride(primals_172, (2048, 512), (512, 1))
    assert_size_stride(primals_173, (2048, ), (1, ))
    assert_size_stride(primals_174, (512, 2048), (2048, 1))
    assert_size_stride(primals_175, (512, ), (1, ))
    assert_size_stride(primals_176, (1536, 512), (512, 1))
    assert_size_stride(primals_177, (1536, ), (1, ))
    assert_size_stride(primals_178, (512, 512), (512, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (2048, 512), (512, 1))
    assert_size_stride(primals_181, (2048, ), (1, ))
    assert_size_stride(primals_182, (512, 2048), (2048, 1))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (1536, 512), (512, 1))
    assert_size_stride(primals_185, (1536, ), (1, ))
    assert_size_stride(primals_186, (512, 512), (512, 1))
    assert_size_stride(primals_187, (512, ), (1, ))
    assert_size_stride(primals_188, (2048, 512), (512, 1))
    assert_size_stride(primals_189, (2048, ), (1, ))
    assert_size_stride(primals_190, (512, 2048), (2048, 1))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (1536, 512), (512, 1))
    assert_size_stride(primals_193, (1536, ), (1, ))
    assert_size_stride(primals_194, (512, 512), (512, 1))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (2048, 512), (512, 1))
    assert_size_stride(primals_197, (2048, ), (1, ))
    assert_size_stride(primals_198, (512, 2048), (2048, 1))
    assert_size_stride(primals_199, (512, ), (1, ))
    assert_size_stride(primals_200, (1536, 512), (512, 1))
    assert_size_stride(primals_201, (1536, ), (1, ))
    assert_size_stride(primals_202, (512, 512), (512, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (2048, 512), (512, 1))
    assert_size_stride(primals_205, (2048, ), (1, ))
    assert_size_stride(primals_206, (512, 2048), (2048, 1))
    assert_size_stride(primals_207, (512, ), (1, ))
    assert_size_stride(primals_208, (1536, 512), (512, 1))
    assert_size_stride(primals_209, (1536, ), (1, ))
    assert_size_stride(primals_210, (512, 512), (512, 1))
    assert_size_stride(primals_211, (512, ), (1, ))
    assert_size_stride(primals_212, (2048, 512), (512, 1))
    assert_size_stride(primals_213, (2048, ), (1, ))
    assert_size_stride(primals_214, (512, 2048), (2048, 1))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (1536, 512), (512, 1))
    assert_size_stride(primals_217, (1536, ), (1, ))
    assert_size_stride(primals_218, (512, 512), (512, 1))
    assert_size_stride(primals_219, (512, ), (1, ))
    assert_size_stride(primals_220, (2048, 512), (512, 1))
    assert_size_stride(primals_221, (2048, ), (1, ))
    assert_size_stride(primals_222, (512, 2048), (2048, 1))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (1536, 512), (512, 1))
    assert_size_stride(primals_225, (1536, ), (1, ))
    assert_size_stride(primals_226, (512, 512), (512, 1))
    assert_size_stride(primals_227, (512, ), (1, ))
    assert_size_stride(primals_228, (2048, 512), (512, 1))
    assert_size_stride(primals_229, (2048, ), (1, ))
    assert_size_stride(primals_230, (512, 2048), (2048, 1))
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (1536, 512), (512, 1))
    assert_size_stride(primals_233, (1536, ), (1, ))
    assert_size_stride(primals_234, (512, 512), (512, 1))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_236, (2048, 512), (512, 1))
    assert_size_stride(primals_237, (2048, ), (1, ))
    assert_size_stride(primals_238, (512, 2048), (2048, 1))
    assert_size_stride(primals_239, (512, ), (1, ))
    assert_size_stride(primals_240, (1536, 512), (512, 1))
    assert_size_stride(primals_241, (1536, ), (1, ))
    assert_size_stride(primals_242, (512, 512), (512, 1))
    assert_size_stride(primals_243, (512, ), (1, ))
    assert_size_stride(primals_244, (2048, 512), (512, 1))
    assert_size_stride(primals_245, (2048, ), (1, ))
    assert_size_stride(primals_246, (512, 2048), (2048, 1))
    assert_size_stride(primals_247, (512, ), (1, ))
    assert_size_stride(primals_248, (1536, 512), (512, 1))
    assert_size_stride(primals_249, (1536, ), (1, ))
    assert_size_stride(primals_250, (512, 512), (512, 1))
    assert_size_stride(primals_251, (512, ), (1, ))
    assert_size_stride(primals_252, (2048, 512), (512, 1))
    assert_size_stride(primals_253, (2048, ), (1, ))
    assert_size_stride(primals_254, (512, 2048), (2048, 1))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (1536, 512), (512, 1))
    assert_size_stride(primals_257, (1536, ), (1, ))
    assert_size_stride(primals_258, (512, 512), (512, 1))
    assert_size_stride(primals_259, (512, ), (1, ))
    assert_size_stride(primals_260, (2048, 512), (512, 1))
    assert_size_stride(primals_261, (2048, ), (1, ))
    assert_size_stride(primals_262, (512, 2048), (2048, 1))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_264, (1536, 512), (512, 1))
    assert_size_stride(primals_265, (1536, ), (1, ))
    assert_size_stride(primals_266, (512, 512), (512, 1))
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (2048, 512), (512, 1))
    assert_size_stride(primals_269, (2048, ), (1, ))
    assert_size_stride(primals_270, (512, 2048), (2048, 1))
    assert_size_stride(primals_271, (512, ), (1, ))
    assert_size_stride(primals_272, (1536, 512), (512, 1))
    assert_size_stride(primals_273, (1536, ), (1, ))
    assert_size_stride(primals_274, (512, 512), (512, 1))
    assert_size_stride(primals_275, (512, ), (1, ))
    assert_size_stride(primals_276, (2048, 512), (512, 1))
    assert_size_stride(primals_277, (2048, ), (1, ))
    assert_size_stride(primals_278, (512, 2048), (2048, 1))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (1536, 512), (512, 1))
    assert_size_stride(primals_281, (1536, ), (1, ))
    assert_size_stride(primals_282, (512, 512), (512, 1))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_284, (2048, 512), (512, 1))
    assert_size_stride(primals_285, (2048, ), (1, ))
    assert_size_stride(primals_286, (512, 2048), (2048, 1))
    assert_size_stride(primals_287, (512, ), (1, ))
    assert_size_stride(primals_288, (1536, 512), (512, 1))
    assert_size_stride(primals_289, (1536, ), (1, ))
    assert_size_stride(primals_290, (512, 512), (512, 1))
    assert_size_stride(primals_291, (512, ), (1, ))
    assert_size_stride(primals_292, (2048, 512), (512, 1))
    assert_size_stride(primals_293, (2048, ), (1, ))
    assert_size_stride(primals_294, (512, 2048), (2048, 1))
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (1536, 512), (512, 1))
    assert_size_stride(primals_297, (1536, ), (1, ))
    assert_size_stride(primals_298, (512, 512), (512, 1))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (2048, 512), (512, 1))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (512, 2048), (2048, 1))
    assert_size_stride(primals_303, (512, ), (1, ))
    assert_size_stride(primals_304, (1000, 512), (512, 1))
    assert_size_stride(primals_305, (1000, ), (1, ))
    assert_size_stride(primals_306, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_106.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del primals_106
    del primals_124
    del primals_142
    del primals_306
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, buf0, primals_107, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf4, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del primals_107
    buf5 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 16, 196, 128), (401408, 1, 2048, 16), device='cpu', dtype=torch.float32)
    buf9 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_1(c_void_p(buf4.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_3
    buf10 = empty((25088, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_109, buf9, reinterpret_tensor(primals_108, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf10)
    del primals_109
    buf11 = empty((8, 4, 16, 196, 32), device='cpu', dtype=torch.float32)
    buf12 = empty((8, 4, 16, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_2(c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    buf13 = empty((512, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf11, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf12, (512, 32, 196), (6272, 196, 1), 0), out=buf13)
    buf14 = empty_strided((8, 4, 16, 196, 1), (12544, 3136, 196, 1, 100352), device='cpu', dtype=torch.float32)
    buf15 = reinterpret_tensor(buf13, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), 0); del buf13  # reuse
    buf16 = empty_strided((8, 4, 16, 196, 1), (12544, 3136, 196, 1, 100352), device='cpu', dtype=torch.float32)
    buf17 = buf15; del buf15  # reuse
    buf18 = empty((8, 4, 16, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_3(c_void_p(buf17.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = empty((512, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf17, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf18, (512, 196, 32), (6272, 32, 1), 0), out=buf19)
    buf20 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_view_4(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    buf21 = reinterpret_tensor(buf19, (25088, 128), (128, 1), 0); del buf19  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf20, reinterpret_tensor(primals_110, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf21)
    del primals_111
    buf22 = buf5; del buf5  # reuse
    buf23 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((8, 16, 196, 128), (401408, 1, 2048, 16), device='cpu', dtype=torch.float32)
    buf26 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_5(c_void_p(buf4.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_5
    buf27 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_113, buf26, reinterpret_tensor(primals_112, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf27)
    del primals_113
    buf28 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_6(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = empty((25088, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_115, buf28, reinterpret_tensor(primals_114, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf29)
    del primals_115
    buf30 = buf22; del buf22  # reuse
    buf31 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((8, 16, 196, 128), (401408, 1, 2048, 16), device='cpu', dtype=torch.float32)
    buf34 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf4.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    del primals_7
    buf35 = buf10; del buf10  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___0___transformer_encoder___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_117, buf34, reinterpret_tensor(primals_116, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf35)
    del primals_117
    buf36 = empty((8, 4, 16, 196, 32), device='cpu', dtype=torch.float32)
    buf37 = empty((8, 4, 16, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_8(c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = empty((512, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf36, (512, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf37, (512, 32, 196), (6272, 196, 1), 0), out=buf38)
    buf39 = buf16; del buf16  # reuse
    buf40 = reinterpret_tensor(buf38, (8, 4, 16, 196, 196), (2458624, 614656, 38416, 196, 1), 0); del buf38  # reuse
    buf41 = buf14; del buf14  # reuse
    buf42 = buf40; del buf40  # reuse
    buf43 = empty((8, 4, 16, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_9(c_void_p(buf42.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()))
    del buf35
    del buf39
    del buf41
    buf44 = empty((512, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf42, (512, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf43, (512, 196, 32), (6272, 32, 1), 0), out=buf44)
    buf45 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_view_10(c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()))
    buf46 = reinterpret_tensor(buf44, (25088, 128), (128, 1), 0); del buf44  # reuse
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, buf45, reinterpret_tensor(primals_118, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf46)
    del primals_119
    buf48 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_11(c_void_p(buf48.data_ptr()))
    aten.bernoulli_(buf48, 0.9782608691602945)
    buf51 = reinterpret_tensor(buf46, (8, 16, 196, 128), (401408, 25088, 128, 1), 0); del buf46  # reuse
    buf52 = buf30; del buf30  # reuse
    buf53 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf55 = empty_strided((8, 16, 196, 128), (401408, 1, 2048, 16), device='cpu', dtype=torch.float32)
    buf56 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_12(c_void_p(buf51.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_1
    del primals_9
    buf57 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_121, buf56, reinterpret_tensor(primals_120, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf57)
    del primals_121
    buf58 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_13(c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = reinterpret_tensor(buf4, (25088, 128), (128, 1), 0); del buf4  # reuse
    # Source Nodes: [x_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_123, buf58, reinterpret_tensor(primals_122, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf59)
    del primals_123
    buf60 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_14(c_void_p(buf60.data_ptr()))
    aten.bernoulli_(buf60, 0.9782608691602945)
    buf63 = reinterpret_tensor(buf29, (8, 128, 56, 56), (401408, 1, 7168, 128), 0); del buf29  # reuse
    cpp_fused_permute_15(c_void_p(buf51.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf63.data_ptr()))
    # Source Nodes: [x_41], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, buf1, primals_125, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf64, (8, 256, 56, 56), (802816, 1, 14336, 256))
    del primals_125
    buf65 = reinterpret_tensor(buf52, (8, 56, 56, 1), (3136, 1, 56, 56), 0); del buf52  # reuse
    buf66 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf68 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf69 = empty_strided((8, 256, 57, 57), (831744, 1, 14592, 256), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    buf71 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.int64)
    buf72 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf75 = empty_strided((8, 4, 196, 256), (200704, 1, 1024, 4), device='cpu', dtype=torch.float32)
    buf76 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_view_16(c_void_p(buf64.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del primals_11
    del primals_14
    buf77 = empty((6272, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___levels___1___transformer_encoder___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_127, buf76, reinterpret_tensor(primals_126, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf77)
    del primals_127
    buf78 = empty((8, 8, 4, 196, 32), device='cpu', dtype=torch.float32)
    buf79 = empty((8, 8, 4, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_17(c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = empty((256, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf78, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf79, (256, 32, 196), (6272, 196, 1), 0), out=buf80)
    buf81 = empty_strided((8, 8, 4, 196, 1), (6272, 784, 196, 1, 50176), device='cpu', dtype=torch.float32)
    buf82 = reinterpret_tensor(buf80, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), 0); del buf80  # reuse
    buf83 = empty_strided((8, 8, 4, 196, 1), (6272, 784, 196, 1, 50176), device='cpu', dtype=torch.float32)
    buf84 = buf82; del buf82  # reuse
    buf85 = empty((8, 8, 4, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_18(c_void_p(buf84.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = empty((256, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf85, (256, 196, 32), (6272, 32, 1), 0), out=buf86)
    buf87 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_view_19(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()))
    buf88 = reinterpret_tensor(buf86, (6272, 256), (256, 1), 0); del buf86  # reuse
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_129, buf87, reinterpret_tensor(primals_128, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf88)
    del primals_129
    buf89 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_20(c_void_p(buf89.data_ptr()))
    aten.bernoulli_(buf89, 0.9565217383205891)
    buf92 = buf72; del buf72  # reuse
    buf93 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf95 = empty_strided((8, 4, 196, 256), (200704, 1, 1024, 4), device='cpu', dtype=torch.float32)
    buf96 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_21(c_void_p(buf70.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del primals_16
    buf97 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_131, buf96, reinterpret_tensor(primals_130, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf97)
    del primals_131
    buf98 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_22(c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    buf99 = empty((6272, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_133, buf98, reinterpret_tensor(primals_132, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf99)
    del primals_133
    buf100 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_23(c_void_p(buf100.data_ptr()))
    aten.bernoulli_(buf100, 0.9565217383205891)
    buf103 = reinterpret_tensor(buf99, (8, 4, 196, 256), (200704, 50176, 256, 1), 0); del buf99  # reuse
    buf104 = buf92; del buf92  # reuse
    buf105 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf107 = empty_strided((8, 4, 196, 256), (200704, 1, 1024, 4), device='cpu', dtype=torch.float32)
    buf108 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_24(c_void_p(buf103.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    del primals_12
    del primals_18
    buf109 = buf77; del buf77  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___1___transformer_encoder___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_135, buf108, reinterpret_tensor(primals_134, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf109)
    del primals_135
    buf110 = reinterpret_tensor(buf88, (8, 8, 4, 196, 32), (200704, 25088, 6272, 32, 1), 0); del buf88  # reuse
    buf111 = reinterpret_tensor(buf70, (8, 8, 4, 32, 196), (200704, 25088, 6272, 196, 1), 0); del buf70  # reuse
    cpp_fused_clone_mul_25(c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = empty((256, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_68], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf110, (256, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf111, (256, 32, 196), (6272, 196, 1), 0), out=buf112)
    buf113 = buf83; del buf83  # reuse
    buf114 = reinterpret_tensor(buf112, (8, 8, 4, 196, 196), (1229312, 153664, 38416, 196, 1), 0); del buf112  # reuse
    buf115 = buf81; del buf81  # reuse
    buf116 = buf114; del buf114  # reuse
    buf117 = empty((8, 8, 4, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_26(c_void_p(buf116.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()))
    del buf109
    del buf113
    del buf115
    buf118 = empty((256, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_68], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf116, (256, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf117, (256, 196, 32), (6272, 32, 1), 0), out=buf118)
    buf119 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_view_27(c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = reinterpret_tensor(buf118, (6272, 256), (256, 1), 0); del buf118  # reuse
    # Source Nodes: [x_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_137, buf119, reinterpret_tensor(primals_136, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf120)
    del primals_137
    buf121 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_28(c_void_p(buf121.data_ptr()))
    aten.bernoulli_(buf121, 0.9347826093435287)
    buf124 = buf104; del buf104  # reuse
    buf125 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf127 = empty_strided((8, 4, 196, 256), (200704, 1, 1024, 4), device='cpu', dtype=torch.float32)
    buf128 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_29(c_void_p(buf103.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    del primals_20
    buf129 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf128, reinterpret_tensor(primals_138, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf129)
    del primals_139
    buf130 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_30(c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    buf131 = empty((6272, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_141, buf130, reinterpret_tensor(primals_140, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf131)
    del primals_141
    buf132 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_31(c_void_p(buf132.data_ptr()))
    aten.bernoulli_(buf132, 0.9347826093435287)
    buf135 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused_permute_32(c_void_p(buf103.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf135.data_ptr()))
    del buf103
    del buf120
    del buf131
    # Source Nodes: [x_85], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(buf135, buf2, primals_143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf136, (8, 512, 28, 28), (401408, 1, 14336, 512))
    del primals_143
    buf137 = reinterpret_tensor(buf124, (8, 28, 28, 1), (784, 1, 28, 28), 0); del buf124  # reuse
    buf138 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf140 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf141 = empty_strided((8, 512, 29, 29), (430592, 1, 14848, 512), device='cpu', dtype=torch.float32)
    buf142 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    buf143 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.int64)
    buf144 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf147 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf148 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_constant_pad_nd_max_pool2d_with_indices_native_layer_norm_view_33(c_void_p(buf136.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del primals_22
    del primals_25
    buf149 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_145, buf148, reinterpret_tensor(primals_144, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf149)
    del primals_145
    buf150 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    buf151 = empty((8, 16, 1, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_34(c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    buf152 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_98], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf150, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf151, (128, 32, 196), (6272, 196, 1), 0), out=buf152)
    buf153 = reinterpret_tensor(buf66, (8, 16, 1, 196, 1), (3136, 196, 25088, 1, 25088), 0); del buf66  # reuse
    buf154 = reinterpret_tensor(buf152, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf152  # reuse
    buf155 = empty_strided((8, 16, 1, 196, 1), (3136, 196, 25088, 1, 25088), device='cpu', dtype=torch.float32)
    buf156 = reinterpret_tensor(buf154, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf154  # reuse
    buf157 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_35(c_void_p(buf156.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()))
    buf158 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_98], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf156, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf157, (128, 196, 32), (6272, 32, 1), 0), out=buf158)
    buf159 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_36(c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()))
    buf160 = reinterpret_tensor(buf158, (1568, 512), (512, 1), 0); del buf158  # reuse
    # Source Nodes: [x_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_147, buf159, reinterpret_tensor(primals_146, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf160)
    del primals_147
    buf161 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_37(c_void_p(buf161.data_ptr()))
    aten.bernoulli_(buf161, 0.9130434766411781)
    buf164 = buf144; del buf144  # reuse
    buf165 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf167 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf168 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_38(c_void_p(buf142.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    del primals_27
    buf169 = reinterpret_tensor(buf59, (1568, 2048), (2048, 1), 0); del buf59  # reuse
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf168, reinterpret_tensor(primals_148, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf169)
    del primals_149
    buf170 = reinterpret_tensor(buf51, (1568, 2048), (2048, 1), 0); del buf51  # reuse
    cpp_fused_gelu_view_39(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()))
    buf171 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_151, buf170, reinterpret_tensor(primals_150, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf171)
    del primals_151
    buf172 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_40(c_void_p(buf172.data_ptr()))
    aten.bernoulli_(buf172, 0.9130434766411781)
    buf175 = reinterpret_tensor(buf171, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf171  # reuse
    buf176 = buf164; del buf164  # reuse
    buf177 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf179 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf180 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_41(c_void_p(buf175.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()))
    del primals_23
    del primals_29
    buf181 = buf149; del buf149  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_153, buf180, reinterpret_tensor(primals_152, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf181)
    del primals_153
    buf182 = reinterpret_tensor(buf160, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf160  # reuse
    buf183 = reinterpret_tensor(buf142, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf142  # reuse
    cpp_fused_clone_mul_42(c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf182, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf183, (128, 32, 196), (6272, 196, 1), 0), out=buf184)
    buf185 = buf155; del buf155  # reuse
    buf186 = reinterpret_tensor(buf184, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf184  # reuse
    buf187 = buf153; del buf153  # reuse
    buf188 = reinterpret_tensor(buf186, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf186  # reuse
    buf189 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_43(c_void_p(buf188.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf189, (128, 196, 32), (6272, 32, 1), 0), out=buf190)
    buf191 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_44(c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    buf192 = reinterpret_tensor(buf190, (1568, 512), (512, 1), 0); del buf190  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_155, buf191, reinterpret_tensor(primals_154, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf192)
    del primals_155
    buf193 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_45(c_void_p(buf193.data_ptr()))
    aten.bernoulli_(buf193, 0.8913043439388275)
    buf196 = buf176; del buf176  # reuse
    buf197 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf199 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf200 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_46(c_void_p(buf175.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del primals_31
    buf201 = reinterpret_tensor(buf21, (1568, 2048), (2048, 1), 0); del buf21  # reuse
    # Source Nodes: [x_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_157, buf200, reinterpret_tensor(primals_156, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf201)
    del primals_157
    buf202 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_47(c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_159, buf202, reinterpret_tensor(primals_158, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf203)
    del primals_159
    buf204 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_48(c_void_p(buf204.data_ptr()))
    aten.bernoulli_(buf204, 0.8913043439388275)
    buf207 = reinterpret_tensor(buf203, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf203  # reuse
    buf208 = buf196; del buf196  # reuse
    buf209 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf212 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_49(c_void_p(buf207.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    del primals_33
    buf213 = buf181; del buf181  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_161, buf212, reinterpret_tensor(primals_160, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf213)
    del primals_161
    buf214 = reinterpret_tensor(buf192, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf192  # reuse
    buf215 = reinterpret_tensor(buf175, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf175  # reuse
    cpp_fused_clone_mul_50(c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf214, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf215, (128, 32, 196), (6272, 196, 1), 0), out=buf216)
    buf217 = buf187; del buf187  # reuse
    buf218 = reinterpret_tensor(buf216, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf216  # reuse
    buf219 = buf185; del buf185  # reuse
    buf220 = reinterpret_tensor(buf218, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf218  # reuse
    buf221 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_51(c_void_p(buf220.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf220, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf221, (128, 196, 32), (6272, 32, 1), 0), out=buf222)
    buf223 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_52(c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    buf224 = reinterpret_tensor(buf222, (1568, 512), (512, 1), 0); del buf222  # reuse
    # Source Nodes: [x_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_163, buf223, reinterpret_tensor(primals_162, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf224)
    del primals_163
    buf225 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_53(c_void_p(buf225.data_ptr()))
    aten.bernoulli_(buf225, 0.8695652186870575)
    buf228 = buf208; del buf208  # reuse
    buf229 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf231 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf232 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_54(c_void_p(buf207.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    del primals_35
    buf233 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_165, buf232, reinterpret_tensor(primals_164, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf233)
    del primals_165
    buf234 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_55(c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_136], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_167, buf234, reinterpret_tensor(primals_166, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf235)
    del primals_167
    buf236 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_56(c_void_p(buf236.data_ptr()))
    aten.bernoulli_(buf236, 0.8695652186870575)
    buf239 = reinterpret_tensor(buf235, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf235  # reuse
    buf240 = buf228; del buf228  # reuse
    buf241 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf243 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf244 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_57(c_void_p(buf239.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()))
    del primals_37
    buf245 = buf213; del buf213  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_169, buf244, reinterpret_tensor(primals_168, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf245)
    del primals_169
    buf246 = reinterpret_tensor(buf224, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf224  # reuse
    buf247 = reinterpret_tensor(buf207, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf207  # reuse
    cpp_fused_clone_mul_58(c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()))
    buf248 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_140], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf246, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf247, (128, 32, 196), (6272, 196, 1), 0), out=buf248)
    buf249 = buf219; del buf219  # reuse
    buf250 = reinterpret_tensor(buf248, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf248  # reuse
    buf251 = buf217; del buf217  # reuse
    buf252 = reinterpret_tensor(buf250, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf250  # reuse
    buf253 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_59(c_void_p(buf252.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()))
    buf254 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_140], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf252, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf253, (128, 196, 32), (6272, 32, 1), 0), out=buf254)
    buf255 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_60(c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()))
    buf256 = reinterpret_tensor(buf254, (1568, 512), (512, 1), 0); del buf254  # reuse
    # Source Nodes: [x_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_171, buf255, reinterpret_tensor(primals_170, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf256)
    del primals_171
    buf257 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_61(c_void_p(buf257.data_ptr()))
    aten.bernoulli_(buf257, 0.8478260785341263)
    buf260 = buf240; del buf240  # reuse
    buf261 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf263 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf264 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_62(c_void_p(buf239.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    del primals_39
    buf265 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_173, buf264, reinterpret_tensor(primals_172, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf265)
    del primals_173
    buf266 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_63(c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_175, buf266, reinterpret_tensor(primals_174, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf267)
    del primals_175
    buf268 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_64(c_void_p(buf268.data_ptr()))
    aten.bernoulli_(buf268, 0.8478260785341263)
    buf271 = reinterpret_tensor(buf267, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf267  # reuse
    buf272 = buf260; del buf260  # reuse
    buf273 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf275 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf276 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_65(c_void_p(buf271.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del primals_41
    buf277 = buf245; del buf245  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___4___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_177, buf276, reinterpret_tensor(primals_176, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf277)
    del primals_177
    buf278 = reinterpret_tensor(buf256, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf256  # reuse
    buf279 = reinterpret_tensor(buf239, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf239  # reuse
    cpp_fused_clone_mul_66(c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    buf280 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_154], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf278, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf279, (128, 32, 196), (6272, 196, 1), 0), out=buf280)
    buf281 = buf251; del buf251  # reuse
    buf282 = reinterpret_tensor(buf280, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf280  # reuse
    buf283 = buf249; del buf249  # reuse
    buf284 = reinterpret_tensor(buf282, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf282  # reuse
    buf285 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_67(c_void_p(buf284.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_154], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf285, (128, 196, 32), (6272, 32, 1), 0), out=buf286)
    buf287 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_68(c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    buf288 = reinterpret_tensor(buf286, (1568, 512), (512, 1), 0); del buf286  # reuse
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_179, buf287, reinterpret_tensor(primals_178, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf288)
    del primals_179
    buf289 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_69(c_void_p(buf289.data_ptr()))
    aten.bernoulli_(buf289, 0.8260869532823563)
    buf292 = buf272; del buf272  # reuse
    buf293 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf296 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_70(c_void_p(buf271.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    del primals_43
    buf297 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_181, buf296, reinterpret_tensor(primals_180, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf297)
    del primals_181
    buf298 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_71(c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    buf299 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_164], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_183, buf298, reinterpret_tensor(primals_182, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf299)
    del primals_183
    buf300 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_72(c_void_p(buf300.data_ptr()))
    aten.bernoulli_(buf300, 0.8260869532823563)
    buf303 = reinterpret_tensor(buf299, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf299  # reuse
    buf304 = buf292; del buf292  # reuse
    buf305 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf307 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf308 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_73(c_void_p(buf303.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del primals_45
    buf309 = buf277; del buf277  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___5___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_185, buf308, reinterpret_tensor(primals_184, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf309)
    del primals_185
    buf310 = reinterpret_tensor(buf288, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf288  # reuse
    buf311 = reinterpret_tensor(buf271, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf271  # reuse
    cpp_fused_clone_mul_74(c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    buf312 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_168], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf310, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf311, (128, 32, 196), (6272, 196, 1), 0), out=buf312)
    buf313 = buf283; del buf283  # reuse
    buf314 = reinterpret_tensor(buf312, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf312  # reuse
    buf315 = buf281; del buf281  # reuse
    buf316 = reinterpret_tensor(buf314, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf314  # reuse
    buf317 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_75(c_void_p(buf316.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()))
    buf318 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_168], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf316, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf317, (128, 196, 32), (6272, 32, 1), 0), out=buf318)
    buf319 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_76(c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    buf320 = reinterpret_tensor(buf318, (1568, 512), (512, 1), 0); del buf318  # reuse
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_187, buf319, reinterpret_tensor(primals_186, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf320)
    del primals_187
    buf321 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_77(c_void_p(buf321.data_ptr()))
    aten.bernoulli_(buf321, 0.8043478280305862)
    buf324 = buf304; del buf304  # reuse
    buf325 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf327 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf328 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_78(c_void_p(buf303.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    del primals_47
    buf329 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_189, buf328, reinterpret_tensor(primals_188, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf329)
    del primals_189
    buf330 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_79(c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    buf331 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_191, buf330, reinterpret_tensor(primals_190, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf331)
    del primals_191
    buf332 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_80(c_void_p(buf332.data_ptr()))
    aten.bernoulli_(buf332, 0.8043478280305862)
    buf335 = reinterpret_tensor(buf331, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf331  # reuse
    buf336 = buf324; del buf324  # reuse
    buf337 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf339 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf340 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_81(c_void_p(buf335.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del primals_49
    buf341 = buf309; del buf309  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___6___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_193, buf340, reinterpret_tensor(primals_192, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf341)
    del primals_193
    buf342 = reinterpret_tensor(buf320, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf320  # reuse
    buf343 = reinterpret_tensor(buf303, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf303  # reuse
    cpp_fused_clone_mul_82(c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    buf344 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_182], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf342, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf343, (128, 32, 196), (6272, 196, 1), 0), out=buf344)
    buf345 = buf315; del buf315  # reuse
    buf346 = reinterpret_tensor(buf344, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf344  # reuse
    buf347 = buf313; del buf313  # reuse
    buf348 = reinterpret_tensor(buf346, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf346  # reuse
    buf349 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_83(c_void_p(buf348.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf349.data_ptr()))
    buf350 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_182], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf348, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf349, (128, 196, 32), (6272, 32, 1), 0), out=buf350)
    buf351 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_84(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    buf352 = reinterpret_tensor(buf350, (1568, 512), (512, 1), 0); del buf350  # reuse
    # Source Nodes: [x_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_195, buf351, reinterpret_tensor(primals_194, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf352)
    del primals_195
    buf353 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_85(c_void_p(buf353.data_ptr()))
    aten.bernoulli_(buf353, 0.782608687877655)
    buf356 = buf336; del buf336  # reuse
    buf357 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf360 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_86(c_void_p(buf335.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf360.data_ptr()))
    del primals_51
    buf361 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_197, buf360, reinterpret_tensor(primals_196, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf361)
    del primals_197
    buf362 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_87(c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_192], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_199, buf362, reinterpret_tensor(primals_198, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf363)
    del primals_199
    buf364 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_88(c_void_p(buf364.data_ptr()))
    aten.bernoulli_(buf364, 0.782608687877655)
    buf367 = reinterpret_tensor(buf363, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf363  # reuse
    buf368 = buf356; del buf356  # reuse
    buf369 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf371 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf372 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_89(c_void_p(buf367.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    del primals_53
    buf373 = buf341; del buf341  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___7___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_201, buf372, reinterpret_tensor(primals_200, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf373)
    del primals_201
    buf374 = reinterpret_tensor(buf352, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf352  # reuse
    buf375 = reinterpret_tensor(buf335, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf335  # reuse
    cpp_fused_clone_mul_90(c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    buf376 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_196], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf374, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf375, (128, 32, 196), (6272, 196, 1), 0), out=buf376)
    buf377 = buf347; del buf347  # reuse
    buf378 = reinterpret_tensor(buf376, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf376  # reuse
    buf379 = buf345; del buf345  # reuse
    buf380 = reinterpret_tensor(buf378, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf378  # reuse
    buf381 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_91(c_void_p(buf380.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_196], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf380, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf381, (128, 196, 32), (6272, 32, 1), 0), out=buf382)
    buf383 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_92(c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    buf384 = reinterpret_tensor(buf382, (1568, 512), (512, 1), 0); del buf382  # reuse
    # Source Nodes: [x_198], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_203, buf383, reinterpret_tensor(primals_202, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf384)
    del primals_203
    buf385 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_93(c_void_p(buf385.data_ptr()))
    aten.bernoulli_(buf385, 0.760869562625885)
    buf388 = buf368; del buf368  # reuse
    buf389 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf391 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf392 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_94(c_void_p(buf367.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()))
    del primals_55
    buf393 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_202], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_205, buf392, reinterpret_tensor(primals_204, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf393)
    del primals_205
    buf394 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_95(c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    buf395 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_206], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_207, buf394, reinterpret_tensor(primals_206, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf395)
    del primals_207
    buf396 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_96(c_void_p(buf396.data_ptr()))
    aten.bernoulli_(buf396, 0.760869562625885)
    buf399 = reinterpret_tensor(buf395, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf395  # reuse
    buf400 = buf388; del buf388  # reuse
    buf401 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf403 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf404 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_97(c_void_p(buf399.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del primals_57
    buf405 = buf373; del buf373  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___8___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_209, buf404, reinterpret_tensor(primals_208, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf405)
    del primals_209
    buf406 = reinterpret_tensor(buf384, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf384  # reuse
    buf407 = reinterpret_tensor(buf367, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf367  # reuse
    cpp_fused_clone_mul_98(c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_210], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf406, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf407, (128, 32, 196), (6272, 196, 1), 0), out=buf408)
    buf409 = buf379; del buf379  # reuse
    buf410 = reinterpret_tensor(buf408, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf408  # reuse
    buf411 = buf377; del buf377  # reuse
    buf412 = reinterpret_tensor(buf410, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf410  # reuse
    buf413 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_99(c_void_p(buf412.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf413.data_ptr()))
    buf414 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_210], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf412, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf413, (128, 196, 32), (6272, 32, 1), 0), out=buf414)
    buf415 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_100(c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()))
    buf416 = reinterpret_tensor(buf414, (1568, 512), (512, 1), 0); del buf414  # reuse
    # Source Nodes: [x_212], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_211, buf415, reinterpret_tensor(primals_210, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf416)
    del primals_211
    buf417 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_101(c_void_p(buf417.data_ptr()))
    aten.bernoulli_(buf417, 0.739130437374115)
    buf420 = buf400; del buf400  # reuse
    buf421 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf423 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf424 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_102(c_void_p(buf399.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()))
    del primals_59
    buf425 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_216], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_213, buf424, reinterpret_tensor(primals_212, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf425)
    del primals_213
    buf426 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_103(c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    buf427 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_220], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_215, buf426, reinterpret_tensor(primals_214, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf427)
    del primals_215
    buf428 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_104(c_void_p(buf428.data_ptr()))
    aten.bernoulli_(buf428, 0.739130437374115)
    buf431 = reinterpret_tensor(buf427, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf427  # reuse
    buf432 = buf420; del buf420  # reuse
    buf433 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf435 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf436 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_105(c_void_p(buf431.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()))
    del primals_61
    buf437 = buf405; del buf405  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___9___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_217, buf436, reinterpret_tensor(primals_216, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf437)
    del primals_217
    buf438 = reinterpret_tensor(buf416, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf416  # reuse
    buf439 = reinterpret_tensor(buf399, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf399  # reuse
    cpp_fused_clone_mul_106(c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()))
    buf440 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_224], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf438, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf439, (128, 32, 196), (6272, 196, 1), 0), out=buf440)
    buf441 = buf411; del buf411  # reuse
    buf442 = reinterpret_tensor(buf440, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf440  # reuse
    buf443 = buf409; del buf409  # reuse
    buf444 = reinterpret_tensor(buf442, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf442  # reuse
    buf445 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_107(c_void_p(buf444.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf445.data_ptr()))
    buf446 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_224], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf444, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf445, (128, 196, 32), (6272, 32, 1), 0), out=buf446)
    buf447 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_108(c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    buf448 = reinterpret_tensor(buf446, (1568, 512), (512, 1), 0); del buf446  # reuse
    # Source Nodes: [x_226], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_219, buf447, reinterpret_tensor(primals_218, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf448)
    del primals_219
    buf449 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_109(c_void_p(buf449.data_ptr()))
    aten.bernoulli_(buf449, 0.717391312122345)
    buf452 = buf432; del buf432  # reuse
    buf453 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf455 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf456 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_110(c_void_p(buf431.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    del primals_63
    buf457 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_230], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_221, buf456, reinterpret_tensor(primals_220, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf457)
    del primals_221
    buf458 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_111(c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    buf459 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_234], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_223, buf458, reinterpret_tensor(primals_222, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf459)
    del primals_223
    buf460 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_112(c_void_p(buf460.data_ptr()))
    aten.bernoulli_(buf460, 0.717391312122345)
    buf463 = reinterpret_tensor(buf459, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf459  # reuse
    buf464 = buf452; del buf452  # reuse
    buf465 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf467 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf468 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_113(c_void_p(buf463.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()))
    del primals_65
    buf469 = buf437; del buf437  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___10___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_225, buf468, reinterpret_tensor(primals_224, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf469)
    del primals_225
    buf470 = reinterpret_tensor(buf448, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf448  # reuse
    buf471 = reinterpret_tensor(buf431, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf431  # reuse
    cpp_fused_clone_mul_114(c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    buf472 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_238], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf470, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf471, (128, 32, 196), (6272, 196, 1), 0), out=buf472)
    buf473 = buf443; del buf443  # reuse
    buf474 = reinterpret_tensor(buf472, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf472  # reuse
    buf475 = buf441; del buf441  # reuse
    buf476 = reinterpret_tensor(buf474, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf474  # reuse
    buf477 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_115(c_void_p(buf476.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()))
    buf478 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_238], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf476, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf477, (128, 196, 32), (6272, 32, 1), 0), out=buf478)
    buf479 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_116(c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()))
    buf480 = reinterpret_tensor(buf478, (1568, 512), (512, 1), 0); del buf478  # reuse
    # Source Nodes: [x_240], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_227, buf479, reinterpret_tensor(primals_226, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf480)
    del primals_227
    buf481 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_117(c_void_p(buf481.data_ptr()))
    aten.bernoulli_(buf481, 0.695652186870575)
    buf484 = buf464; del buf464  # reuse
    buf485 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf487 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf488 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_118(c_void_p(buf463.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()))
    del primals_67
    buf489 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_244], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_229, buf488, reinterpret_tensor(primals_228, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf489)
    del primals_229
    buf490 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_119(c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()))
    buf491 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_248], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_231, buf490, reinterpret_tensor(primals_230, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf491)
    del primals_231
    buf492 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_120(c_void_p(buf492.data_ptr()))
    aten.bernoulli_(buf492, 0.695652186870575)
    buf495 = reinterpret_tensor(buf491, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf491  # reuse
    buf496 = buf484; del buf484  # reuse
    buf497 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf499 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf500 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_121(c_void_p(buf495.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del primals_69
    buf501 = buf469; del buf469  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___11___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_233, buf500, reinterpret_tensor(primals_232, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf501)
    del primals_233
    buf502 = reinterpret_tensor(buf480, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf480  # reuse
    buf503 = reinterpret_tensor(buf463, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf463  # reuse
    cpp_fused_clone_mul_122(c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()))
    buf504 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_252], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf502, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf503, (128, 32, 196), (6272, 196, 1), 0), out=buf504)
    buf505 = buf475; del buf475  # reuse
    buf506 = reinterpret_tensor(buf504, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf504  # reuse
    buf507 = buf473; del buf473  # reuse
    buf508 = reinterpret_tensor(buf506, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf506  # reuse
    buf509 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_123(c_void_p(buf508.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf509.data_ptr()))
    buf510 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_252], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf508, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf509, (128, 196, 32), (6272, 32, 1), 0), out=buf510)
    buf511 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_124(c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()))
    buf512 = reinterpret_tensor(buf510, (1568, 512), (512, 1), 0); del buf510  # reuse
    # Source Nodes: [x_254], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_235, buf511, reinterpret_tensor(primals_234, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf512)
    del primals_235
    buf513 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_125(c_void_p(buf513.data_ptr()))
    aten.bernoulli_(buf513, 0.6739130616188049)
    buf516 = buf496; del buf496  # reuse
    buf517 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf519 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf520 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_126(c_void_p(buf495.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf520.data_ptr()))
    del primals_71
    buf521 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_258], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_237, buf520, reinterpret_tensor(primals_236, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf521)
    del primals_237
    buf522 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_127(c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()))
    buf523 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_262], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_239, buf522, reinterpret_tensor(primals_238, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf523)
    del primals_239
    buf524 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_128(c_void_p(buf524.data_ptr()))
    aten.bernoulli_(buf524, 0.6739130616188049)
    buf527 = reinterpret_tensor(buf523, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf523  # reuse
    buf528 = buf516; del buf516  # reuse
    buf529 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf531 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf532 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_129(c_void_p(buf527.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    del primals_73
    buf533 = buf501; del buf501  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___12___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_241, buf532, reinterpret_tensor(primals_240, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf533)
    del primals_241
    buf534 = reinterpret_tensor(buf512, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf512  # reuse
    buf535 = reinterpret_tensor(buf495, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf495  # reuse
    cpp_fused_clone_mul_130(c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()))
    buf536 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_266], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf534, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf535, (128, 32, 196), (6272, 196, 1), 0), out=buf536)
    buf537 = buf507; del buf507  # reuse
    buf538 = reinterpret_tensor(buf536, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf536  # reuse
    buf539 = buf505; del buf505  # reuse
    buf540 = reinterpret_tensor(buf538, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf538  # reuse
    buf541 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_131(c_void_p(buf540.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf541.data_ptr()))
    buf542 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_266], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf540, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf541, (128, 196, 32), (6272, 32, 1), 0), out=buf542)
    buf543 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_132(c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()))
    buf544 = reinterpret_tensor(buf542, (1568, 512), (512, 1), 0); del buf542  # reuse
    # Source Nodes: [x_268], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_243, buf543, reinterpret_tensor(primals_242, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf544)
    del primals_243
    buf545 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_133(c_void_p(buf545.data_ptr()))
    aten.bernoulli_(buf545, 0.6521739065647125)
    buf548 = buf528; del buf528  # reuse
    buf549 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf551 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf552 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_134(c_void_p(buf527.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf552.data_ptr()))
    del primals_75
    buf553 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_272], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_245, buf552, reinterpret_tensor(primals_244, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf553)
    del primals_245
    buf554 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_135(c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()))
    buf555 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_276], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_247, buf554, reinterpret_tensor(primals_246, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf555)
    del primals_247
    buf556 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_136(c_void_p(buf556.data_ptr()))
    aten.bernoulli_(buf556, 0.6521739065647125)
    buf559 = reinterpret_tensor(buf555, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf555  # reuse
    buf560 = buf548; del buf548  # reuse
    buf561 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf563 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf564 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_137(c_void_p(buf559.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()))
    del primals_77
    buf565 = buf533; del buf533  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___13___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_249, buf564, reinterpret_tensor(primals_248, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf565)
    del primals_249
    buf566 = reinterpret_tensor(buf544, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf544  # reuse
    buf567 = reinterpret_tensor(buf527, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf527  # reuse
    cpp_fused_clone_mul_138(c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()))
    buf568 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_280], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf566, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf567, (128, 32, 196), (6272, 196, 1), 0), out=buf568)
    buf569 = buf539; del buf539  # reuse
    buf570 = reinterpret_tensor(buf568, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf568  # reuse
    buf571 = buf537; del buf537  # reuse
    buf572 = reinterpret_tensor(buf570, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf570  # reuse
    buf573 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_139(c_void_p(buf572.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf573.data_ptr()))
    buf574 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_280], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf572, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf573, (128, 196, 32), (6272, 32, 1), 0), out=buf574)
    buf575 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_140(c_void_p(buf574.data_ptr()), c_void_p(buf575.data_ptr()))
    buf576 = reinterpret_tensor(buf574, (1568, 512), (512, 1), 0); del buf574  # reuse
    # Source Nodes: [x_282], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_251, buf575, reinterpret_tensor(primals_250, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf576)
    del primals_251
    buf577 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_141(c_void_p(buf577.data_ptr()))
    aten.bernoulli_(buf577, 0.6304347813129425)
    buf580 = buf560; del buf560  # reuse
    buf581 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf583 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf584 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_142(c_void_p(buf559.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()))
    del primals_79
    buf585 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_286], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_253, buf584, reinterpret_tensor(primals_252, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf585)
    del primals_253
    buf586 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_143(c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()))
    buf587 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_290], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_255, buf586, reinterpret_tensor(primals_254, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf587)
    del primals_255
    buf588 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_144(c_void_p(buf588.data_ptr()))
    aten.bernoulli_(buf588, 0.6304347813129425)
    buf591 = reinterpret_tensor(buf587, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf587  # reuse
    buf592 = buf580; del buf580  # reuse
    buf593 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf595 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf596 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_145(c_void_p(buf591.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()))
    del primals_81
    buf597 = buf565; del buf565  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___14___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_257, buf596, reinterpret_tensor(primals_256, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf597)
    del primals_257
    buf598 = reinterpret_tensor(buf576, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf576  # reuse
    buf599 = reinterpret_tensor(buf559, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf559  # reuse
    cpp_fused_clone_mul_146(c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()))
    buf600 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_294], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf598, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf599, (128, 32, 196), (6272, 196, 1), 0), out=buf600)
    buf601 = buf571; del buf571  # reuse
    buf602 = reinterpret_tensor(buf600, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf600  # reuse
    buf603 = buf569; del buf569  # reuse
    buf604 = reinterpret_tensor(buf602, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf602  # reuse
    buf605 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_147(c_void_p(buf604.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf605.data_ptr()))
    buf606 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_294], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf604, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf605, (128, 196, 32), (6272, 32, 1), 0), out=buf606)
    buf607 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_148(c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()))
    buf608 = reinterpret_tensor(buf606, (1568, 512), (512, 1), 0); del buf606  # reuse
    # Source Nodes: [x_296], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_259, buf607, reinterpret_tensor(primals_258, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf608)
    del primals_259
    buf609 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_149(c_void_p(buf609.data_ptr()))
    aten.bernoulli_(buf609, 0.6086956560611725)
    buf612 = buf592; del buf592  # reuse
    buf613 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf615 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf616 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_150(c_void_p(buf591.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf616.data_ptr()))
    del primals_83
    buf617 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_300], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_261, buf616, reinterpret_tensor(primals_260, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf617)
    del primals_261
    buf618 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_151(c_void_p(buf617.data_ptr()), c_void_p(buf618.data_ptr()))
    buf619 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_304], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_263, buf618, reinterpret_tensor(primals_262, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf619)
    del primals_263
    buf620 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_152(c_void_p(buf620.data_ptr()))
    aten.bernoulli_(buf620, 0.6086956560611725)
    buf623 = reinterpret_tensor(buf619, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf619  # reuse
    buf624 = buf612; del buf612  # reuse
    buf625 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf627 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf628 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_153(c_void_p(buf623.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf628.data_ptr()))
    del primals_85
    buf629 = buf597; del buf597  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___15___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_265, buf628, reinterpret_tensor(primals_264, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf629)
    del primals_265
    buf630 = reinterpret_tensor(buf608, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf608  # reuse
    buf631 = reinterpret_tensor(buf591, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf591  # reuse
    cpp_fused_clone_mul_154(c_void_p(buf629.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()))
    buf632 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_308], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf630, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf631, (128, 32, 196), (6272, 196, 1), 0), out=buf632)
    buf633 = buf603; del buf603  # reuse
    buf634 = reinterpret_tensor(buf632, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf632  # reuse
    buf635 = buf601; del buf601  # reuse
    buf636 = reinterpret_tensor(buf634, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf634  # reuse
    buf637 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_155(c_void_p(buf636.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf637.data_ptr()))
    buf638 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_308], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf636, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf637, (128, 196, 32), (6272, 32, 1), 0), out=buf638)
    buf639 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_156(c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()))
    buf640 = reinterpret_tensor(buf638, (1568, 512), (512, 1), 0); del buf638  # reuse
    # Source Nodes: [x_310], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_267, buf639, reinterpret_tensor(primals_266, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf640)
    del primals_267
    buf641 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_157(c_void_p(buf641.data_ptr()))
    aten.bernoulli_(buf641, 0.5869565308094025)
    buf644 = buf624; del buf624  # reuse
    buf645 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf647 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf648 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_158(c_void_p(buf623.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()))
    del primals_87
    buf649 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_314], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_269, buf648, reinterpret_tensor(primals_268, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf649)
    del primals_269
    buf650 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_159(c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()))
    buf651 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_318], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_271, buf650, reinterpret_tensor(primals_270, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf651)
    del primals_271
    buf652 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_160(c_void_p(buf652.data_ptr()))
    aten.bernoulli_(buf652, 0.5869565308094025)
    buf655 = reinterpret_tensor(buf651, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf651  # reuse
    buf656 = buf644; del buf644  # reuse
    buf657 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf659 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf660 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_161(c_void_p(buf655.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf660.data_ptr()))
    del primals_89
    buf661 = buf629; del buf629  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___16___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_273, buf660, reinterpret_tensor(primals_272, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf661)
    del primals_273
    buf662 = reinterpret_tensor(buf640, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf640  # reuse
    buf663 = reinterpret_tensor(buf623, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf623  # reuse
    cpp_fused_clone_mul_162(c_void_p(buf661.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()))
    buf664 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_322], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf662, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf663, (128, 32, 196), (6272, 196, 1), 0), out=buf664)
    buf665 = buf635; del buf635  # reuse
    buf666 = reinterpret_tensor(buf664, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf664  # reuse
    buf667 = buf633; del buf633  # reuse
    buf668 = reinterpret_tensor(buf666, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf666  # reuse
    buf669 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_163(c_void_p(buf668.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf669.data_ptr()))
    buf670 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_322], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf668, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf669, (128, 196, 32), (6272, 32, 1), 0), out=buf670)
    buf671 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_164(c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()))
    buf672 = reinterpret_tensor(buf670, (1568, 512), (512, 1), 0); del buf670  # reuse
    # Source Nodes: [x_324], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_275, buf671, reinterpret_tensor(primals_274, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf672)
    del primals_275
    buf673 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_165(c_void_p(buf673.data_ptr()))
    aten.bernoulli_(buf673, 0.5652174055576324)
    buf676 = buf656; del buf656  # reuse
    buf677 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf679 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf680 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_166(c_void_p(buf655.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()))
    del primals_91
    buf681 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_328], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_277, buf680, reinterpret_tensor(primals_276, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf681)
    del primals_277
    buf682 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_167(c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()))
    buf683 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_332], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_279, buf682, reinterpret_tensor(primals_278, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf683)
    del primals_279
    buf684 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_168(c_void_p(buf684.data_ptr()))
    aten.bernoulli_(buf684, 0.5652174055576324)
    buf687 = reinterpret_tensor(buf683, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf683  # reuse
    buf688 = buf676; del buf676  # reuse
    buf689 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf691 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf692 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_169(c_void_p(buf687.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf692.data_ptr()))
    del primals_93
    buf693 = buf661; del buf661  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___17___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_281, buf692, reinterpret_tensor(primals_280, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf693)
    del primals_281
    buf694 = reinterpret_tensor(buf672, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf672  # reuse
    buf695 = reinterpret_tensor(buf655, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf655  # reuse
    cpp_fused_clone_mul_170(c_void_p(buf693.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()))
    buf696 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_336], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf694, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf695, (128, 32, 196), (6272, 196, 1), 0), out=buf696)
    buf697 = buf667; del buf667  # reuse
    buf698 = reinterpret_tensor(buf696, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf696  # reuse
    buf699 = buf665; del buf665  # reuse
    buf700 = reinterpret_tensor(buf698, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf698  # reuse
    buf701 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_171(c_void_p(buf700.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf699.data_ptr()), c_void_p(buf701.data_ptr()))
    buf702 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_336], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf700, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf701, (128, 196, 32), (6272, 32, 1), 0), out=buf702)
    buf703 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_172(c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()))
    buf704 = reinterpret_tensor(buf702, (1568, 512), (512, 1), 0); del buf702  # reuse
    # Source Nodes: [x_338], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_283, buf703, reinterpret_tensor(primals_282, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf704)
    del primals_283
    buf705 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_173(c_void_p(buf705.data_ptr()))
    aten.bernoulli_(buf705, 0.54347825050354)
    buf708 = buf688; del buf688  # reuse
    buf709 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf711 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf712 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_174(c_void_p(buf687.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf712.data_ptr()))
    del primals_95
    buf713 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_342], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_285, buf712, reinterpret_tensor(primals_284, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf713)
    del primals_285
    buf714 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_175(c_void_p(buf713.data_ptr()), c_void_p(buf714.data_ptr()))
    buf715 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_346], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_287, buf714, reinterpret_tensor(primals_286, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf715)
    del primals_287
    buf716 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_176(c_void_p(buf716.data_ptr()))
    aten.bernoulli_(buf716, 0.54347825050354)
    buf719 = reinterpret_tensor(buf715, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf715  # reuse
    buf720 = buf708; del buf708  # reuse
    buf721 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf723 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf724 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_177(c_void_p(buf719.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf724.data_ptr()))
    del primals_97
    buf725 = buf693; del buf693  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___18___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_289, buf724, reinterpret_tensor(primals_288, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf725)
    del primals_289
    buf726 = reinterpret_tensor(buf704, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf704  # reuse
    buf727 = reinterpret_tensor(buf687, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf687  # reuse
    cpp_fused_clone_mul_178(c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf727.data_ptr()))
    buf728 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_350], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf726, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf727, (128, 32, 196), (6272, 196, 1), 0), out=buf728)
    buf729 = buf699; del buf699  # reuse
    buf730 = reinterpret_tensor(buf728, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf728  # reuse
    buf731 = buf697; del buf697  # reuse
    buf732 = reinterpret_tensor(buf730, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf730  # reuse
    buf733 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_179(c_void_p(buf732.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf729.data_ptr()), c_void_p(buf731.data_ptr()), c_void_p(buf733.data_ptr()))
    buf734 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_350], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf732, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf733, (128, 196, 32), (6272, 32, 1), 0), out=buf734)
    buf735 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_180(c_void_p(buf734.data_ptr()), c_void_p(buf735.data_ptr()))
    buf736 = reinterpret_tensor(buf734, (1568, 512), (512, 1), 0); del buf734  # reuse
    # Source Nodes: [x_352], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_291, buf735, reinterpret_tensor(primals_290, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf736)
    del primals_291
    buf737 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_181(c_void_p(buf737.data_ptr()))
    aten.bernoulli_(buf737, 0.52173912525177)
    buf740 = buf720; del buf720  # reuse
    buf741 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf743 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf744 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_182(c_void_p(buf719.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf740.data_ptr()), c_void_p(buf741.data_ptr()), c_void_p(buf743.data_ptr()), c_void_p(buf744.data_ptr()))
    del primals_99
    buf745 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_356], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_293, buf744, reinterpret_tensor(primals_292, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf745)
    del primals_293
    buf746 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_183(c_void_p(buf745.data_ptr()), c_void_p(buf746.data_ptr()))
    buf747 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_360], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_295, buf746, reinterpret_tensor(primals_294, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf747)
    del primals_295
    buf748 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_184(c_void_p(buf748.data_ptr()))
    aten.bernoulli_(buf748, 0.52173912525177)
    buf751 = reinterpret_tensor(buf747, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf747  # reuse
    buf752 = buf740; del buf740  # reuse
    buf753 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf755 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf756 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_185(c_void_p(buf751.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf737.data_ptr()), c_void_p(buf748.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(buf753.data_ptr()), c_void_p(buf755.data_ptr()), c_void_p(buf756.data_ptr()))
    del primals_101
    buf757 = buf725; del buf725  # reuse
    # Source Nodes: [getattr_getattr_l__mod___levels___2___transformer_encoder___19___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_297, buf756, reinterpret_tensor(primals_296, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf757)
    del primals_297
    buf758 = reinterpret_tensor(buf736, (8, 16, 1, 196, 32), (100352, 6272, 6272, 32, 1), 0); del buf736  # reuse
    buf759 = reinterpret_tensor(buf719, (8, 16, 1, 32, 196), (100352, 6272, 6272, 196, 1), 0); del buf719  # reuse
    cpp_fused_clone_mul_186(c_void_p(buf757.data_ptr()), c_void_p(buf758.data_ptr()), c_void_p(buf759.data_ptr()))
    buf760 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_364], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf758, (128, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf759, (128, 32, 196), (6272, 196, 1), 0), out=buf760)
    buf761 = buf731; del buf731  # reuse
    buf762 = reinterpret_tensor(buf760, (8, 16, 1, 196, 196), (614656, 38416, 4917248, 196, 1), 0); del buf760  # reuse
    buf763 = buf729; del buf729  # reuse
    buf764 = reinterpret_tensor(buf762, (8, 16, 1, 196, 196), (614656, 38416, 38416, 196, 1), 0); del buf762  # reuse
    buf765 = empty((8, 16, 1, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_187(c_void_p(buf764.data_ptr()), c_void_p(buf757.data_ptr()), c_void_p(buf761.data_ptr()), c_void_p(buf763.data_ptr()), c_void_p(buf765.data_ptr()))
    del buf757
    buf766 = empty((128, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_364], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf764, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf765, (128, 196, 32), (6272, 32, 1), 0), out=buf766)
    buf767 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_view_188(c_void_p(buf766.data_ptr()), c_void_p(buf767.data_ptr()))
    buf768 = reinterpret_tensor(buf766, (1568, 512), (512, 1), 0); del buf766  # reuse
    # Source Nodes: [x_366], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_299, buf767, reinterpret_tensor(primals_298, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf768)
    del primals_299
    buf769 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_189(c_void_p(buf769.data_ptr()))
    aten.bernoulli_(buf769, 0.5)
    buf772 = buf752; del buf752  # reuse
    buf773 = empty_strided((8, 1, 196, 1), (196, 1568, 1, 1568), device='cpu', dtype=torch.float32)
    buf775 = empty_strided((8, 1, 196, 512), (100352, 1, 512, 1), device='cpu', dtype=torch.float32)
    buf776 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_view_190(c_void_p(buf751.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf772.data_ptr()), c_void_p(buf773.data_ptr()), c_void_p(buf775.data_ptr()), c_void_p(buf776.data_ptr()))
    del primals_103
    buf777 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_370], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_301, buf776, reinterpret_tensor(primals_300, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf777)
    del primals_301
    buf778 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_191(c_void_p(buf777.data_ptr()), c_void_p(buf778.data_ptr()))
    buf779 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_374], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_303, buf778, reinterpret_tensor(primals_302, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf779)
    del primals_303
    buf780 = empty((8, 1, 1, 1), device='cpu', dtype=torch.float32)
    cpp_fused_bernoulli_192(c_void_p(buf780.data_ptr()))
    aten.bernoulli_(buf780, 0.5)
    buf783 = reinterpret_tensor(buf779, (8, 1, 196, 512), (100352, 802816, 512, 1), 0); del buf779  # reuse
    buf784 = reinterpret_tensor(buf772, (8, 14, 14, 1), (196, 14, 1, 1568), 0); del buf772  # reuse
    buf785 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf787 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cpu', dtype=torch.float32)
    buf788 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cpu', dtype=torch.float32)
    buf789 = reinterpret_tensor(buf788, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf788  # reuse
    cpp_fused_add_div_mean_mul_native_layer_norm_193(c_void_p(buf783.data_ptr()), c_void_p(buf789.data_ptr()), c_void_p(buf751.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf769.data_ptr()), c_void_p(buf780.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf787.data_ptr()))
    del buf751
    del buf768
    del buf783
    del primals_105
    buf790 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_305, reinterpret_tensor(buf789, (8, 512), (512, 1), 0), reinterpret_tensor(primals_304, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf790)
    del primals_305
    buf791 = reinterpret_tensor(buf784, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf784  # reuse
    buf792 = reinterpret_tensor(buf773, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf773  # reuse
    buf793 = reinterpret_tensor(buf753, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf753  # reuse
    buf794 = reinterpret_tensor(buf741, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf741  # reuse
    buf795 = reinterpret_tensor(buf721, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf721  # reuse
    buf796 = reinterpret_tensor(buf709, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf709  # reuse
    buf797 = reinterpret_tensor(buf689, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf689  # reuse
    buf798 = reinterpret_tensor(buf677, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf677  # reuse
    buf799 = reinterpret_tensor(buf657, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf657  # reuse
    buf800 = reinterpret_tensor(buf645, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf645  # reuse
    buf801 = reinterpret_tensor(buf625, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf625  # reuse
    buf802 = reinterpret_tensor(buf613, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf613  # reuse
    buf803 = reinterpret_tensor(buf593, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf593  # reuse
    buf804 = reinterpret_tensor(buf581, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf581  # reuse
    buf805 = reinterpret_tensor(buf561, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf561  # reuse
    buf806 = reinterpret_tensor(buf549, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf549  # reuse
    buf807 = reinterpret_tensor(buf529, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf529  # reuse
    buf808 = reinterpret_tensor(buf517, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf517  # reuse
    buf809 = reinterpret_tensor(buf497, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf497  # reuse
    buf810 = reinterpret_tensor(buf485, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf485  # reuse
    buf811 = reinterpret_tensor(buf465, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf465  # reuse
    buf812 = reinterpret_tensor(buf453, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf453  # reuse
    buf813 = reinterpret_tensor(buf433, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf433  # reuse
    buf814 = reinterpret_tensor(buf421, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf421  # reuse
    buf815 = reinterpret_tensor(buf401, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf401  # reuse
    buf816 = reinterpret_tensor(buf389, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf389  # reuse
    buf817 = reinterpret_tensor(buf369, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf369  # reuse
    buf818 = reinterpret_tensor(buf357, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf357  # reuse
    buf819 = reinterpret_tensor(buf337, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf337  # reuse
    buf820 = reinterpret_tensor(buf325, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf325  # reuse
    buf821 = reinterpret_tensor(buf305, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf305  # reuse
    buf822 = reinterpret_tensor(buf293, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf293  # reuse
    buf823 = reinterpret_tensor(buf273, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf273  # reuse
    buf824 = reinterpret_tensor(buf261, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf261  # reuse
    buf825 = reinterpret_tensor(buf241, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf241  # reuse
    buf826 = reinterpret_tensor(buf229, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf229  # reuse
    buf827 = reinterpret_tensor(buf209, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf209  # reuse
    buf828 = reinterpret_tensor(buf197, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf197  # reuse
    buf829 = reinterpret_tensor(buf177, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf177  # reuse
    buf830 = reinterpret_tensor(buf165, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf165  # reuse
    buf831 = reinterpret_tensor(buf145, (8, 1, 196, 1), (196, 1, 1, 1), 0); del buf145  # reuse
    buf832 = reinterpret_tensor(buf138, (8, 4, 196, 1), (784, 1, 4, 4), 0); del buf138  # reuse
    buf833 = empty_strided((8, 4, 196, 1), (784, 1, 4, 4), device='cpu', dtype=torch.float32)
    buf834 = empty_strided((8, 4, 196, 1), (784, 1, 4, 4), device='cpu', dtype=torch.float32)
    buf835 = empty_strided((8, 4, 196, 1), (784, 1, 4, 4), device='cpu', dtype=torch.float32)
    buf836 = reinterpret_tensor(buf763, (8, 16, 196, 1), (3136, 1, 16, 16), 0); del buf763  # reuse
    buf837 = reinterpret_tensor(buf761, (8, 16, 196, 1), (3136, 1, 16, 16), 0); del buf761  # reuse
    buf838 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf839 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mul_native_layer_norm_native_layer_norm_backward_194(c_void_p(buf792.data_ptr()), c_void_p(buf793.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(buf795.data_ptr()), c_void_p(buf796.data_ptr()), c_void_p(buf797.data_ptr()), c_void_p(buf798.data_ptr()), c_void_p(buf799.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf801.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf803.data_ptr()), c_void_p(buf804.data_ptr()), c_void_p(buf805.data_ptr()), c_void_p(buf806.data_ptr()), c_void_p(buf807.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf809.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf811.data_ptr()), c_void_p(buf812.data_ptr()), c_void_p(buf813.data_ptr()), c_void_p(buf814.data_ptr()), c_void_p(buf815.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf817.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf819.data_ptr()), c_void_p(buf820.data_ptr()), c_void_p(buf821.data_ptr()), c_void_p(buf822.data_ptr()), c_void_p(buf823.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf825.data_ptr()), c_void_p(buf826.data_ptr()), c_void_p(buf827.data_ptr()), c_void_p(buf828.data_ptr()), c_void_p(buf829.data_ptr()), c_void_p(buf830.data_ptr()), c_void_p(buf831.data_ptr()), c_void_p(buf785.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf791.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(buf833.data_ptr()), c_void_p(buf834.data_ptr()), c_void_p(buf835.data_ptr()), c_void_p(buf836.data_ptr()), c_void_p(buf837.data_ptr()), c_void_p(buf838.data_ptr()), c_void_p(buf839.data_ptr()))
    return (buf790, primals_2, primals_4, primals_6, primals_8, primals_10, primals_13, primals_15, primals_17, primals_19, primals_21, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, buf0, buf1, buf2, buf3, buf8, buf9, buf20, buf25, buf26, buf27, buf28, buf33, buf34, buf45, buf48, buf55, buf56, buf57, buf58, buf60, buf63, buf64, buf65, buf68, buf69, buf71, buf75, buf76, buf87, buf89, buf95, buf96, buf97, buf98, buf100, buf107, buf108, buf119, buf121, buf127, buf128, buf129, buf130, buf132, buf135, buf136, buf137, buf140, buf141, buf143, buf147, buf148, buf159, buf161, buf167, buf168, buf169, buf170, buf172, buf179, buf180, buf191, buf193, buf199, buf200, buf201, buf202, buf204, buf211, buf212, buf223, buf225, buf231, buf232, buf233, buf234, buf236, buf243, buf244, buf255, buf257, buf263, buf264, buf265, buf266, buf268, buf275, buf276, buf287, buf289, buf295, buf296, buf297, buf298, buf300, buf307, buf308, buf319, buf321, buf327, buf328, buf329, buf330, buf332, buf339, buf340, buf351, buf353, buf359, buf360, buf361, buf362, buf364, buf371, buf372, buf383, buf385, buf391, buf392, buf393, buf394, buf396, buf403, buf404, buf415, buf417, buf423, buf424, buf425, buf426, buf428, buf435, buf436, buf447, buf449, buf455, buf456, buf457, buf458, buf460, buf467, buf468, buf479, buf481, buf487, buf488, buf489, buf490, buf492, buf499, buf500, buf511, buf513, buf519, buf520, buf521, buf522, buf524, buf531, buf532, buf543, buf545, buf551, buf552, buf553, buf554, buf556, buf563, buf564, buf575, buf577, buf583, buf584, buf585, buf586, buf588, buf595, buf596, buf607, buf609, buf615, buf616, buf617, buf618, buf620, buf627, buf628, buf639, buf641, buf647, buf648, buf649, buf650, buf652, buf659, buf660, buf671, buf673, buf679, buf680, buf681, buf682, buf684, buf691, buf692, buf703, buf705, buf711, buf712, buf713, buf714, buf716, buf723, buf724, buf735, buf737, buf743, buf744, buf745, buf746, buf748, buf755, buf756, buf767, buf769, buf775, buf776, buf777, buf778, buf780, buf787, reinterpret_tensor(buf789, (8, 512), (512, 1), 0), reinterpret_tensor(primals_304, (1000, 512), (512, 1), 0), buf791, reinterpret_tensor(primals_302, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_300, (2048, 512), (512, 1), 0), buf792, reinterpret_tensor(primals_298, (512, 512), (512, 1), 0), reinterpret_tensor(buf764, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf765, (128, 32, 196), (6272, 1, 32), 0), buf764, reinterpret_tensor(buf758, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf759, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_296, (1536, 512), (512, 1), 0), buf793, reinterpret_tensor(primals_294, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_292, (2048, 512), (512, 1), 0), buf794, reinterpret_tensor(primals_290, (512, 512), (512, 1), 0), reinterpret_tensor(buf732, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf733, (128, 32, 196), (6272, 1, 32), 0), buf732, reinterpret_tensor(buf726, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf727, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_288, (1536, 512), (512, 1), 0), buf795, reinterpret_tensor(primals_286, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_284, (2048, 512), (512, 1), 0), buf796, reinterpret_tensor(primals_282, (512, 512), (512, 1), 0), reinterpret_tensor(buf700, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf701, (128, 32, 196), (6272, 1, 32), 0), buf700, reinterpret_tensor(buf694, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf695, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_280, (1536, 512), (512, 1), 0), buf797, reinterpret_tensor(primals_278, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_276, (2048, 512), (512, 1), 0), buf798, reinterpret_tensor(primals_274, (512, 512), (512, 1), 0), reinterpret_tensor(buf668, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf669, (128, 32, 196), (6272, 1, 32), 0), buf668, reinterpret_tensor(buf662, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf663, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_272, (1536, 512), (512, 1), 0), buf799, reinterpret_tensor(primals_270, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_268, (2048, 512), (512, 1), 0), buf800, reinterpret_tensor(primals_266, (512, 512), (512, 1), 0), reinterpret_tensor(buf636, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf637, (128, 32, 196), (6272, 1, 32), 0), buf636, reinterpret_tensor(buf630, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf631, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_264, (1536, 512), (512, 1), 0), buf801, reinterpret_tensor(primals_262, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_260, (2048, 512), (512, 1), 0), buf802, reinterpret_tensor(primals_258, (512, 512), (512, 1), 0), reinterpret_tensor(buf604, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf605, (128, 32, 196), (6272, 1, 32), 0), buf604, reinterpret_tensor(buf598, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf599, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_256, (1536, 512), (512, 1), 0), buf803, reinterpret_tensor(primals_254, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_252, (2048, 512), (512, 1), 0), buf804, reinterpret_tensor(primals_250, (512, 512), (512, 1), 0), reinterpret_tensor(buf572, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf573, (128, 32, 196), (6272, 1, 32), 0), buf572, reinterpret_tensor(buf566, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf567, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_248, (1536, 512), (512, 1), 0), buf805, reinterpret_tensor(primals_246, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_244, (2048, 512), (512, 1), 0), buf806, reinterpret_tensor(primals_242, (512, 512), (512, 1), 0), reinterpret_tensor(buf540, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf541, (128, 32, 196), (6272, 1, 32), 0), buf540, reinterpret_tensor(buf534, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf535, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_240, (1536, 512), (512, 1), 0), buf807, reinterpret_tensor(primals_238, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_236, (2048, 512), (512, 1), 0), buf808, reinterpret_tensor(primals_234, (512, 512), (512, 1), 0), reinterpret_tensor(buf508, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf509, (128, 32, 196), (6272, 1, 32), 0), buf508, reinterpret_tensor(buf502, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf503, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_232, (1536, 512), (512, 1), 0), buf809, reinterpret_tensor(primals_230, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_228, (2048, 512), (512, 1), 0), buf810, reinterpret_tensor(primals_226, (512, 512), (512, 1), 0), reinterpret_tensor(buf476, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf477, (128, 32, 196), (6272, 1, 32), 0), buf476, reinterpret_tensor(buf470, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf471, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_224, (1536, 512), (512, 1), 0), buf811, reinterpret_tensor(primals_222, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_220, (2048, 512), (512, 1), 0), buf812, reinterpret_tensor(primals_218, (512, 512), (512, 1), 0), reinterpret_tensor(buf444, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf445, (128, 32, 196), (6272, 1, 32), 0), buf444, reinterpret_tensor(buf438, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf439, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_216, (1536, 512), (512, 1), 0), buf813, reinterpret_tensor(primals_214, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_212, (2048, 512), (512, 1), 0), buf814, reinterpret_tensor(primals_210, (512, 512), (512, 1), 0), reinterpret_tensor(buf412, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf413, (128, 32, 196), (6272, 1, 32), 0), buf412, reinterpret_tensor(buf406, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf407, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_208, (1536, 512), (512, 1), 0), buf815, reinterpret_tensor(primals_206, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_204, (2048, 512), (512, 1), 0), buf816, reinterpret_tensor(primals_202, (512, 512), (512, 1), 0), reinterpret_tensor(buf380, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf381, (128, 32, 196), (6272, 1, 32), 0), buf380, reinterpret_tensor(buf374, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf375, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_200, (1536, 512), (512, 1), 0), buf817, reinterpret_tensor(primals_198, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_196, (2048, 512), (512, 1), 0), buf818, reinterpret_tensor(primals_194, (512, 512), (512, 1), 0), reinterpret_tensor(buf348, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf349, (128, 32, 196), (6272, 1, 32), 0), buf348, reinterpret_tensor(buf342, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf343, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_192, (1536, 512), (512, 1), 0), buf819, reinterpret_tensor(primals_190, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_188, (2048, 512), (512, 1), 0), buf820, reinterpret_tensor(primals_186, (512, 512), (512, 1), 0), reinterpret_tensor(buf316, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf317, (128, 32, 196), (6272, 1, 32), 0), buf316, reinterpret_tensor(buf310, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf311, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_184, (1536, 512), (512, 1), 0), buf821, reinterpret_tensor(primals_182, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_180, (2048, 512), (512, 1), 0), buf822, reinterpret_tensor(primals_178, (512, 512), (512, 1), 0), reinterpret_tensor(buf284, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf285, (128, 32, 196), (6272, 1, 32), 0), buf284, reinterpret_tensor(buf278, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf279, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_176, (1536, 512), (512, 1), 0), buf823, reinterpret_tensor(primals_174, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_172, (2048, 512), (512, 1), 0), buf824, reinterpret_tensor(primals_170, (512, 512), (512, 1), 0), reinterpret_tensor(buf252, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf253, (128, 32, 196), (6272, 1, 32), 0), buf252, reinterpret_tensor(buf246, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf247, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_168, (1536, 512), (512, 1), 0), buf825, reinterpret_tensor(primals_166, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_164, (2048, 512), (512, 1), 0), buf826, reinterpret_tensor(primals_162, (512, 512), (512, 1), 0), reinterpret_tensor(buf220, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf221, (128, 32, 196), (6272, 1, 32), 0), buf220, reinterpret_tensor(buf214, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf215, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_160, (1536, 512), (512, 1), 0), buf827, reinterpret_tensor(primals_158, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_156, (2048, 512), (512, 1), 0), buf828, reinterpret_tensor(primals_154, (512, 512), (512, 1), 0), reinterpret_tensor(buf188, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf189, (128, 32, 196), (6272, 1, 32), 0), buf188, reinterpret_tensor(buf182, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf183, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_152, (1536, 512), (512, 1), 0), buf829, reinterpret_tensor(primals_150, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_148, (2048, 512), (512, 1), 0), buf830, reinterpret_tensor(primals_146, (512, 512), (512, 1), 0), reinterpret_tensor(buf156, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf157, (128, 32, 196), (6272, 1, 32), 0), buf156, reinterpret_tensor(buf150, (128, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf151, (128, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_144, (1536, 512), (512, 1), 0), buf831, reinterpret_tensor(primals_140, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_138, (1024, 256), (256, 1), 0), buf832, reinterpret_tensor(primals_136, (256, 256), (256, 1), 0), reinterpret_tensor(buf116, (256, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf117, (256, 32, 196), (6272, 1, 32), 0), buf116, reinterpret_tensor(buf110, (256, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf111, (256, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_134, (768, 256), (256, 1), 0), buf833, reinterpret_tensor(primals_132, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_130, (1024, 256), (256, 1), 0), buf834, reinterpret_tensor(primals_128, (256, 256), (256, 1), 0), reinterpret_tensor(buf84, (256, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf85, (256, 32, 196), (6272, 1, 32), 0), buf84, reinterpret_tensor(buf78, (256, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf79, (256, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_126, (768, 256), (256, 1), 0), buf835, reinterpret_tensor(primals_122, (128, 512), (512, 1), 0), reinterpret_tensor(primals_120, (512, 128), (128, 1), 0), buf836, reinterpret_tensor(primals_118, (128, 128), (128, 1), 0), reinterpret_tensor(buf42, (512, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf43, (512, 32, 196), (6272, 1, 32), 0), buf42, reinterpret_tensor(buf36, (512, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf37, (512, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_116, (384, 128), (128, 1), 0), buf837, reinterpret_tensor(primals_114, (128, 512), (512, 1), 0), reinterpret_tensor(primals_112, (512, 128), (128, 1), 0), buf838, reinterpret_tensor(primals_110, (128, 128), (128, 1), 0), reinterpret_tensor(buf17, (512, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf18, (512, 32, 196), (6272, 1, 32), 0), buf17, reinterpret_tensor(buf11, (512, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf12, (512, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_108, (384, 128), (128, 1), 0), buf839, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 16, 196, 128), (401408, 25088, 128, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1, 4, 196, 256), (200704, 50176, 256, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((1, 1, 196, 512), (100352, 100352, 512, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((1000, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('jx_nest_base', benchmark_compiled_module)
