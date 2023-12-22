
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (512L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1024L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (256L*x2) + (1024L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (2048L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x2) + (2048L*x0)));
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
                        auto tmp0 = in_ptr4[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr4[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_permute_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>(x3 + (128L*x2) + (128L*x2_inner) + (7168L*x1) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (56L*x1) + (3136L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (56L*x1) + (3136L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(128.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-06);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (56L*x3) + (7168L*x2) + (7168L*x2_inner) + (401408L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(x2 + (56L*x1) + (56L*x1_inner) + (7168L*x3) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x3) + (7168L*x2) + (401408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                                auto tmp1 = static_cast<float>(128.0);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((56L*(static_cast<long>(x0) % static_cast<long>(56L))) + (3136L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 56L)) % static_cast<long>(56L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((56L*(static_cast<long>(x0) % static_cast<long>(56L))) + (3136L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 56L)) % static_cast<long>(56L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_3 = async_compile.cpp('''
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


cpp_fused_add_mul_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                                auto tmp1 = static_cast<float>(128.0);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((56L*(static_cast<long>(x0) % static_cast<long>(56L))) + (3136L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 56L)) % static_cast<long>(56L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((56L*(static_cast<long>(x0) % static_cast<long>(56L))) + (3136L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 56L)) % static_cast<long>(56L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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


cpp_fused_add_mul_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                                auto tmp1 = static_cast<float>(128.0);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((56L*(static_cast<long>(x0) % static_cast<long>(56L))) + (3136L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 56L)) % static_cast<long>(56L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((56L*(static_cast<long>(x0) % static_cast<long>(56L))) + (3136L*(c10::div_floor_integer(x0, 3136L))) + (static_cast<long>(c10::div_floor_integer(x0, 56L)) % static_cast<long>(56L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_9 = async_compile.cpp('''
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


cpp_fused_native_layer_norm_permute_10 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp2 = tmp0 * tmp1;
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (128L*x2) + (7168L*x1) + (401408L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (56L*x1) + (3136L*x0))];
                            auto tmp2 = tmp0 * tmp1;
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (56L*x3) + (56L*x3_inner) + (7168L*x2) + (401408L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(x2 + (56L*x1) + (56L*x1_inner) + (7168L*x3) + (401408L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr3 + static_cast<long>(x1 + (128L*x3) + (7168L*x2) + (401408L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(256.0);
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
                            auto tmp1 = static_cast<float>(256.0);
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
                            auto tmp1 = static_cast<float>(256.0);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((28L*(static_cast<long>(x0) % static_cast<long>(28L))) + (784L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((28L*(static_cast<long>(x0) % static_cast<long>(28L))) + (784L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_12 = async_compile.cpp('''
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


cpp_fused_add_mul_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(256.0);
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
                            auto tmp1 = static_cast<float>(256.0);
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
                            auto tmp1 = static_cast<float>(256.0);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((28L*(static_cast<long>(x0) % static_cast<long>(28L))) + (784L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((28L*(static_cast<long>(x0) % static_cast<long>(28L))) + (784L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_15 = async_compile.cpp('''
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


cpp_fused_add_mul_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(256.0);
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
                            auto tmp1 = static_cast<float>(256.0);
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
                            auto tmp1 = static_cast<float>(256.0);
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((28L*(static_cast<long>(x0) % static_cast<long>(28L))) + (784L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((28L*(static_cast<long>(x0) % static_cast<long>(28L))) + (784L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_18 = async_compile.cpp('''
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


cpp_fused_native_layer_norm_permute_19 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp2 = tmp0 * tmp1;
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (256L*x2) + (7168L*x1) + (200704L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp2 = tmp0 * tmp1;
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (7168L*x2) + (200704L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (7168L*x3) + (200704L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr3 + static_cast<long>(x1 + (256L*x3) + (7168L*x2) + (200704L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_21 = async_compile.cpp('''
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


cpp_fused_add_mul_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_24 = async_compile.cpp('''
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


cpp_fused_add_mul_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_27 = async_compile.cpp('''
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


cpp_fused_add_mul_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_mul_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_33 = async_compile.cpp('''
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


cpp_fused_add_mul_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_36 = async_compile.cpp('''
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


cpp_fused_add_mul_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_mul_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_42 = async_compile.cpp('''
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


cpp_fused_add_mul_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_45 = async_compile.cpp('''
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


cpp_fused_add_mul_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_48 = async_compile.cpp('''
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


cpp_fused_add_mul_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_51 = async_compile.cpp('''
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


cpp_fused_add_mul_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_54 = async_compile.cpp('''
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


cpp_fused_add_mul_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_57 = async_compile.cpp('''
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


cpp_fused_add_mul_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_60 = async_compile.cpp('''
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


cpp_fused_add_mul_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_mul_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_66 = async_compile.cpp('''
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


cpp_fused_add_mul_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_69 = async_compile.cpp('''
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


cpp_fused_add_mul_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_72 = async_compile.cpp('''
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


cpp_fused_add_mul_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_75 = async_compile.cpp('''
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


cpp_fused_add_mul_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_78 = async_compile.cpp('''
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


cpp_fused_add_mul_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_81 = async_compile.cpp('''
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


cpp_fused_add_mul_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_84 = async_compile.cpp('''
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


cpp_fused_add_mul_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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


cpp_fused_add_mul_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_90 = async_compile.cpp('''
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


cpp_fused_add_mul_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_93 = async_compile.cpp('''
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


cpp_fused_add_mul_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_96 = async_compile.cpp('''
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


cpp_fused_add_mul_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(512.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-06);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                tmp7.store(tmp8 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp8, 8, out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-06);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            tmp7.store(out_ptr2 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(512.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp5;
                        }
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((14L*(static_cast<long>(x0) % static_cast<long>(14L))) + (196L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_99 = async_compile.cpp('''
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


cpp_fused_native_layer_norm_permute_100 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (512L*x0)));
                        auto tmp2 = tmp0 * tmp1;
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
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (512L*x2) + (7168L*x1) + (100352L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = tmp0 * tmp1;
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (7168L*x2) + (100352L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (7168L*x3) + (100352L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr3 + static_cast<long>(x1 + (512L*x3) + (7168L*x2) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (7L*x2) + (49L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp1 = static_cast<float>(1024.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (7L*x2) + (49L*x0))] = tmp5;
                        }
                    }
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
                    auto tmp1 = out_ptr0[static_cast<long>((7L*(static_cast<long>(x0) % static_cast<long>(7L))) + (49L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(c10::div_floor_integer(x0, 7L)) % static_cast<long>(7L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((7L*(static_cast<long>(x0) % static_cast<long>(7L))) + (49L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(c10::div_floor_integer(x0, 7L)) % static_cast<long>(7L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (7L*x2) + (49L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp1 = static_cast<float>(1024.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (7L*x2) + (49L*x0))] = tmp5;
                        }
                    }
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
                    auto tmp1 = out_ptr0[static_cast<long>((7L*(static_cast<long>(x0) % static_cast<long>(7L))) + (49L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(c10::div_floor_integer(x0, 7L)) % static_cast<long>(7L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((7L*(static_cast<long>(x0) % static_cast<long>(7L))) + (49L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(c10::div_floor_integer(x0, 7L)) % static_cast<long>(7L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mul_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                            #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                            Welford<float> tmp_acc0 = Welford<float>();
                            Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(1024L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (1024L*x2) + (7168L*x1) + (50176L*x0)));
                                tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                            }
                            tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                            out_ptr0[static_cast<long>(x1 + (7L*x2) + (49L*x0))] = static_cast<float>(tmp_acc0.mean);
                        }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr1[static_cast<long>(x2 + (7L*x1) + (49L*x0))];
                            auto tmp1 = static_cast<float>(1024.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-06);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            out_ptr2[static_cast<long>(x1 + (7L*x2) + (49L*x0))] = tmp5;
                        }
                    }
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
                    auto tmp1 = out_ptr0[static_cast<long>((7L*(static_cast<long>(x0) % static_cast<long>(7L))) + (49L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(c10::div_floor_integer(x0, 7L)) % static_cast<long>(7L)))];
                    auto tmp4 = out_ptr2[static_cast<long>((7L*(static_cast<long>(x0) % static_cast<long>(7L))) + (49L*(c10::div_floor_integer(x0, 49L))) + (static_cast<long>(c10::div_floor_integer(x0, 7L)) % static_cast<long>(7L)))];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
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


cpp_fused_add_mean_mul_native_layer_norm_view_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x2) + (50176L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
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
                        out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                        out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr1[static_cast<long>(x0)];
                        auto tmp4 = out_ptr2[static_cast<long>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(1024.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-06);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp11.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                        tmp15.store(out_ptr4 + static_cast<long>(x1 + (1024L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_native_layer_norm_backward_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1024.0);
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                {
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                        auto tmp1 = static_cast<float>(256.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 / tmp2;
                        auto tmp4 = static_cast<float>(1e-06);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 + tmp5;
                        auto tmp7 = tmp6.rsqrt();
                        auto tmp8 = tmp7 / tmp2;
                        tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr1 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr1 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-06);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    auto tmp6 = tmp5 / tmp1;
                    out_ptr1[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
                }
            }
        }
    }
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
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (56L*x1) + (56L*x1_inner) + (3136L*x0)));
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
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr2 + static_cast<long>(x1 + (56L*x2) + (3136L*x0)), static_cast<long>(56L));
                }
            }
        }
    }
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
                    float tmp9[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (56L*x1) + (56L*x1_inner) + (3136L*x0)));
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
                    at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr3 + static_cast<long>(x1 + (56L*x2) + (3136L*x0)), static_cast<long>(56L));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
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
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (1024, ), (1, ))
    assert_size_stride(primals_109, (1024, ), (1, ))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (1024, ), (1, ))
    assert_size_stride(primals_113, (1024, ), (1, ))
    assert_size_stride(primals_114, (1024, ), (1, ))
    assert_size_stride(primals_115, (1024, ), (1, ))
    assert_size_stride(primals_116, (1024, ), (1, ))
    assert_size_stride(primals_117, (1024, ), (1, ))
    assert_size_stride(primals_118, (1024, ), (1, ))
    assert_size_stride(primals_119, (128, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (512, 128), (128, 1))
    assert_size_stride(primals_124, (512, ), (1, ))
    assert_size_stride(primals_125, (128, 512), (512, 1))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (512, 128), (128, 1))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (128, 512), (512, 1))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (512, 128), (128, 1))
    assert_size_stride(primals_136, (512, ), (1, ))
    assert_size_stride(primals_137, (128, 512), (512, 1))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (256, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_143, (1024, 256), (256, 1))
    assert_size_stride(primals_144, (1024, ), (1, ))
    assert_size_stride(primals_145, (256, 1024), (1024, 1))
    assert_size_stride(primals_146, (256, ), (1, ))
    assert_size_stride(primals_147, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_148, (256, ), (1, ))
    assert_size_stride(primals_149, (1024, 256), (256, 1))
    assert_size_stride(primals_150, (1024, ), (1, ))
    assert_size_stride(primals_151, (256, 1024), (1024, 1))
    assert_size_stride(primals_152, (256, ), (1, ))
    assert_size_stride(primals_153, (256, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_154, (256, ), (1, ))
    assert_size_stride(primals_155, (1024, 256), (256, 1))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (256, 1024), (1024, 1))
    assert_size_stride(primals_158, (256, ), (1, ))
    assert_size_stride(primals_159, (512, 256, 2, 2), (1024, 4, 2, 1))
    assert_size_stride(primals_160, (512, ), (1, ))
    assert_size_stride(primals_161, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_162, (512, ), (1, ))
    assert_size_stride(primals_163, (2048, 512), (512, 1))
    assert_size_stride(primals_164, (2048, ), (1, ))
    assert_size_stride(primals_165, (512, 2048), (2048, 1))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_168, (512, ), (1, ))
    assert_size_stride(primals_169, (2048, 512), (512, 1))
    assert_size_stride(primals_170, (2048, ), (1, ))
    assert_size_stride(primals_171, (512, 2048), (2048, 1))
    assert_size_stride(primals_172, (512, ), (1, ))
    assert_size_stride(primals_173, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (2048, 512), (512, 1))
    assert_size_stride(primals_176, (2048, ), (1, ))
    assert_size_stride(primals_177, (512, 2048), (2048, 1))
    assert_size_stride(primals_178, (512, ), (1, ))
    assert_size_stride(primals_179, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (2048, 512), (512, 1))
    assert_size_stride(primals_182, (2048, ), (1, ))
    assert_size_stride(primals_183, (512, 2048), (2048, 1))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (2048, 512), (512, 1))
    assert_size_stride(primals_188, (2048, ), (1, ))
    assert_size_stride(primals_189, (512, 2048), (2048, 1))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_192, (512, ), (1, ))
    assert_size_stride(primals_193, (2048, 512), (512, 1))
    assert_size_stride(primals_194, (2048, ), (1, ))
    assert_size_stride(primals_195, (512, 2048), (2048, 1))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_198, (512, ), (1, ))
    assert_size_stride(primals_199, (2048, 512), (512, 1))
    assert_size_stride(primals_200, (2048, ), (1, ))
    assert_size_stride(primals_201, (512, 2048), (2048, 1))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (2048, 512), (512, 1))
    assert_size_stride(primals_206, (2048, ), (1, ))
    assert_size_stride(primals_207, (512, 2048), (2048, 1))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (2048, 512), (512, 1))
    assert_size_stride(primals_212, (2048, ), (1, ))
    assert_size_stride(primals_213, (512, 2048), (2048, 1))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_216, (512, ), (1, ))
    assert_size_stride(primals_217, (2048, 512), (512, 1))
    assert_size_stride(primals_218, (2048, ), (1, ))
    assert_size_stride(primals_219, (512, 2048), (2048, 1))
    assert_size_stride(primals_220, (512, ), (1, ))
    assert_size_stride(primals_221, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (2048, 512), (512, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_225, (512, 2048), (2048, 1))
    assert_size_stride(primals_226, (512, ), (1, ))
    assert_size_stride(primals_227, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_228, (512, ), (1, ))
    assert_size_stride(primals_229, (2048, 512), (512, 1))
    assert_size_stride(primals_230, (2048, ), (1, ))
    assert_size_stride(primals_231, (512, 2048), (2048, 1))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (2048, 512), (512, 1))
    assert_size_stride(primals_236, (2048, ), (1, ))
    assert_size_stride(primals_237, (512, 2048), (2048, 1))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_239, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (2048, 512), (512, 1))
    assert_size_stride(primals_242, (2048, ), (1, ))
    assert_size_stride(primals_243, (512, 2048), (2048, 1))
    assert_size_stride(primals_244, (512, ), (1, ))
    assert_size_stride(primals_245, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (2048, 512), (512, 1))
    assert_size_stride(primals_248, (2048, ), (1, ))
    assert_size_stride(primals_249, (512, 2048), (2048, 1))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_252, (512, ), (1, ))
    assert_size_stride(primals_253, (2048, 512), (512, 1))
    assert_size_stride(primals_254, (2048, ), (1, ))
    assert_size_stride(primals_255, (512, 2048), (2048, 1))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (2048, 512), (512, 1))
    assert_size_stride(primals_260, (2048, ), (1, ))
    assert_size_stride(primals_261, (512, 2048), (2048, 1))
    assert_size_stride(primals_262, (512, ), (1, ))
    assert_size_stride(primals_263, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (2048, 512), (512, 1))
    assert_size_stride(primals_266, (2048, ), (1, ))
    assert_size_stride(primals_267, (512, 2048), (2048, 1))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (2048, 512), (512, 1))
    assert_size_stride(primals_272, (2048, ), (1, ))
    assert_size_stride(primals_273, (512, 2048), (2048, 1))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (2048, 512), (512, 1))
    assert_size_stride(primals_278, (2048, ), (1, ))
    assert_size_stride(primals_279, (512, 2048), (2048, 1))
    assert_size_stride(primals_280, (512, ), (1, ))
    assert_size_stride(primals_281, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_282, (512, ), (1, ))
    assert_size_stride(primals_283, (2048, 512), (512, 1))
    assert_size_stride(primals_284, (2048, ), (1, ))
    assert_size_stride(primals_285, (512, 2048), (2048, 1))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_288, (512, ), (1, ))
    assert_size_stride(primals_289, (2048, 512), (512, 1))
    assert_size_stride(primals_290, (2048, ), (1, ))
    assert_size_stride(primals_291, (512, 2048), (2048, 1))
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (2048, 512), (512, 1))
    assert_size_stride(primals_296, (2048, ), (1, ))
    assert_size_stride(primals_297, (512, 2048), (2048, 1))
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (2048, 512), (512, 1))
    assert_size_stride(primals_302, (2048, ), (1, ))
    assert_size_stride(primals_303, (512, 2048), (2048, 1))
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_306, (512, ), (1, ))
    assert_size_stride(primals_307, (2048, 512), (512, 1))
    assert_size_stride(primals_308, (2048, ), (1, ))
    assert_size_stride(primals_309, (512, 2048), (2048, 1))
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_312, (512, ), (1, ))
    assert_size_stride(primals_313, (2048, 512), (512, 1))
    assert_size_stride(primals_314, (2048, ), (1, ))
    assert_size_stride(primals_315, (512, 2048), (2048, 1))
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (512, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_318, (512, ), (1, ))
    assert_size_stride(primals_319, (2048, 512), (512, 1))
    assert_size_stride(primals_320, (2048, ), (1, ))
    assert_size_stride(primals_321, (512, 2048), (2048, 1))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (1024, 512, 2, 2), (2048, 4, 2, 1))
    assert_size_stride(primals_324, (1024, ), (1, ))
    assert_size_stride(primals_325, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_326, (1024, ), (1, ))
    assert_size_stride(primals_327, (4096, 1024), (1024, 1))
    assert_size_stride(primals_328, (4096, ), (1, ))
    assert_size_stride(primals_329, (1024, 4096), (4096, 1))
    assert_size_stride(primals_330, (1024, ), (1, ))
    assert_size_stride(primals_331, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_332, (1024, ), (1, ))
    assert_size_stride(primals_333, (4096, 1024), (1024, 1))
    assert_size_stride(primals_334, (4096, ), (1, ))
    assert_size_stride(primals_335, (1024, 4096), (4096, 1))
    assert_size_stride(primals_336, (1024, ), (1, ))
    assert_size_stride(primals_337, (1024, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_338, (1024, ), (1, ))
    assert_size_stride(primals_339, (4096, 1024), (1024, 1))
    assert_size_stride(primals_340, (4096, ), (1, ))
    assert_size_stride(primals_341, (1024, 4096), (4096, 1))
    assert_size_stride(primals_342, (1024, ), (1, ))
    assert_size_stride(primals_343, (1000, 1024), (1024, 1))
    assert_size_stride(primals_344, (1000, ), (1, ))
    assert_size_stride(primals_345, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((128, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((256, 128, 2, 2), (512, 1, 256, 128), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((512, 256, 2, 2), (1024, 1, 512, 256), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((1024, 512, 2, 2), (2048, 1, 1024, 512), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_119.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del primals_119
    del primals_139
    del primals_159
    del primals_323
    del primals_345
    # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf4, buf0, primals_120, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf5, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del primals_120
    buf6 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf9 = empty_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_permute_1(c_void_p(buf5.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    del primals_2
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(buf10, primals_121, primals_122, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf11, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del primals_122
    buf12 = reinterpret_tensor(buf6, (8, 56, 56, 1), (3136, 1, 56, 56), 0); del buf6  # reuse
    buf13 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf5, (25088, 128), (128, 1), 0); del buf5  # reuse
    cpp_fused_native_layer_norm_view_2(c_void_p(buf11.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()))
    del primals_4
    buf17 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf16, reinterpret_tensor(primals_123, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf17)
    del primals_124
    buf18 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_3(c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = empty((25088, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_126, buf18, reinterpret_tensor(primals_125, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf19)
    del primals_126
    buf20 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_4(c_void_p(buf19.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf20.data_ptr()))
    # Source Nodes: [x_19], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(buf20, primals_127, primals_128, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf21, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del primals_128
    buf22 = reinterpret_tensor(buf13, (8, 56, 56, 1), (3136, 1, 56, 56), 0); del buf13  # reuse
    buf23 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf26 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_5(c_void_p(buf21.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_7
    buf27 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf26, reinterpret_tensor(primals_129, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf27)
    del primals_130
    buf28 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_6(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = empty((25088, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf28, reinterpret_tensor(primals_131, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf29)
    del primals_132
    buf30 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_7(c_void_p(buf29.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf30.data_ptr()))
    # Source Nodes: [x_33], Original ATen: [aten.convolution]
    buf31 = extern_kernels.convolution(buf30, primals_133, primals_134, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf31, (8, 128, 56, 56), (401408, 1, 7168, 128))
    del primals_134
    buf32 = reinterpret_tensor(buf23, (8, 56, 56, 1), (3136, 1, 56, 56), 0); del buf23  # reuse
    buf33 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf35 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf36 = empty((25088, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_8(c_void_p(buf31.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    del primals_10
    buf37 = empty((25088, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_136, buf36, reinterpret_tensor(primals_135, (128, 512), (1, 128), 0), alpha=1, beta=1, out=buf37)
    del primals_136
    buf38 = empty((25088, 512), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_9(c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = empty((25088, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf38, reinterpret_tensor(primals_137, (512, 128), (1, 512), 0), alpha=1, beta=1, out=buf39)
    del primals_138
    buf40 = buf33; del buf33  # reuse
    buf41 = empty_strided((8, 56, 56, 1), (3136, 56, 1, 25088), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((8, 56, 56, 128), (401408, 1, 7168, 56), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((8, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_permute_10(c_void_p(buf39.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    del primals_13
    # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
    buf45 = extern_kernels.convolution(buf44, buf1, primals_140, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf45, (8, 256, 28, 28), (200704, 1, 7168, 256))
    del primals_140
    # Source Nodes: [x_52], Original ATen: [aten.convolution]
    buf46 = extern_kernels.convolution(buf45, primals_141, primals_142, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256)
    assert_size_stride(buf46, (8, 256, 28, 28), (200704, 1, 7168, 256))
    del primals_142
    buf47 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf48 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf50 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf51 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_11(c_void_p(buf46.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()))
    del primals_15
    buf52 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf51, reinterpret_tensor(primals_143, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf52)
    del primals_144
    buf53 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_12(c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    buf54 = empty((6272, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf53, reinterpret_tensor(primals_145, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf54)
    del primals_146
    buf55 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_13(c_void_p(buf54.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf55.data_ptr()))
    # Source Nodes: [x_66], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, primals_147, primals_148, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256)
    assert_size_stride(buf56, (8, 256, 28, 28), (200704, 1, 7168, 256))
    del primals_148
    buf57 = reinterpret_tensor(buf48, (8, 28, 28, 1), (784, 1, 28, 28), 0); del buf48  # reuse
    buf58 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf60 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf61 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_14(c_void_p(buf56.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    del primals_18
    buf62 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf61, reinterpret_tensor(primals_149, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf62)
    del primals_150
    buf63 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_15(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    buf64 = empty((6272, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf63, reinterpret_tensor(primals_151, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf64)
    del primals_152
    buf65 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_16(c_void_p(buf64.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf65.data_ptr()))
    # Source Nodes: [x_80], Original ATen: [aten.convolution]
    buf66 = extern_kernels.convolution(buf65, primals_153, primals_154, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256)
    assert_size_stride(buf66, (8, 256, 28, 28), (200704, 1, 7168, 256))
    del primals_154
    buf67 = reinterpret_tensor(buf58, (8, 28, 28, 1), (784, 1, 28, 28), 0); del buf58  # reuse
    buf68 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf71 = empty((6272, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_17(c_void_p(buf66.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del primals_21
    buf72 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf71, reinterpret_tensor(primals_155, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf72)
    del primals_156
    buf73 = empty((6272, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_18(c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()))
    buf74 = empty((6272, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf73, reinterpret_tensor(primals_157, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf74)
    del primals_158
    buf75 = buf68; del buf68  # reuse
    buf76 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((8, 28, 28, 256), (200704, 1, 7168, 28), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((8, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_permute_19(c_void_p(buf74.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del primals_24
    # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
    buf80 = extern_kernels.convolution(buf79, buf2, primals_160, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf80, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_160
    # Source Nodes: [x_99], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, primals_161, primals_162, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf81, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_162
    buf82 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf83 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf85 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf86 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_20(c_void_p(buf81.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del primals_26
    buf87 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_164, buf86, reinterpret_tensor(primals_163, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf87)
    del primals_164
    buf88 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_21(c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    buf89 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf88, reinterpret_tensor(primals_165, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf89)
    del primals_166
    buf90 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_22(c_void_p(buf89.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf90.data_ptr()))
    # Source Nodes: [x_113], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(buf90, primals_167, primals_168, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf91, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_168
    buf92 = reinterpret_tensor(buf83, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf83  # reuse
    buf93 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf95 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf96 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_23(c_void_p(buf91.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del primals_29
    buf97 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_170, buf96, reinterpret_tensor(primals_169, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf97)
    del primals_170
    buf98 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_24(c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    buf99 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_172, buf98, reinterpret_tensor(primals_171, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf99)
    del primals_172
    buf100 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_25(c_void_p(buf99.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf100.data_ptr()))
    # Source Nodes: [x_127], Original ATen: [aten.convolution]
    buf101 = extern_kernels.convolution(buf100, primals_173, primals_174, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf101, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_174
    buf102 = reinterpret_tensor(buf93, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf93  # reuse
    buf103 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf105 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf106 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_26(c_void_p(buf101.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_32
    buf107 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_176, buf106, reinterpret_tensor(primals_175, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf107)
    del primals_176
    buf108 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_27(c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    buf109 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_135], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, buf108, reinterpret_tensor(primals_177, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf109)
    del primals_178
    buf110 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_28(c_void_p(buf109.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf110.data_ptr()))
    # Source Nodes: [x_141], Original ATen: [aten.convolution]
    buf111 = extern_kernels.convolution(buf110, primals_179, primals_180, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf111, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_180
    buf112 = reinterpret_tensor(buf103, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf103  # reuse
    buf113 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf115 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf116 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_29(c_void_p(buf111.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del primals_35
    buf117 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_145], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf116, reinterpret_tensor(primals_181, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf117)
    del primals_182
    buf118 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_30(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()))
    buf119 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_184, buf118, reinterpret_tensor(primals_183, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf119)
    del primals_184
    buf120 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_31(c_void_p(buf119.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf120.data_ptr()))
    # Source Nodes: [x_155], Original ATen: [aten.convolution]
    buf121 = extern_kernels.convolution(buf120, primals_185, primals_186, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf121, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_186
    buf122 = reinterpret_tensor(buf113, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf113  # reuse
    buf123 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf125 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf126 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_32(c_void_p(buf121.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del primals_38
    buf127 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf126, reinterpret_tensor(primals_187, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf127)
    del primals_188
    buf128 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_33(c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()))
    buf129 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_190, buf128, reinterpret_tensor(primals_189, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf129)
    del primals_190
    buf130 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_34(c_void_p(buf129.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf130.data_ptr()))
    # Source Nodes: [x_169], Original ATen: [aten.convolution]
    buf131 = extern_kernels.convolution(buf130, primals_191, primals_192, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf131, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_192
    buf132 = reinterpret_tensor(buf123, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf123  # reuse
    buf133 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf135 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf136 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_35(c_void_p(buf131.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    del primals_41
    buf137 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_194, buf136, reinterpret_tensor(primals_193, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf137)
    del primals_194
    buf138 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_36(c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    buf139 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_196, buf138, reinterpret_tensor(primals_195, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf139)
    del primals_196
    buf140 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_37(c_void_p(buf139.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf140.data_ptr()))
    # Source Nodes: [x_183], Original ATen: [aten.convolution]
    buf141 = extern_kernels.convolution(buf140, primals_197, primals_198, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf141, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_198
    buf142 = reinterpret_tensor(buf133, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf133  # reuse
    buf143 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf145 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf146 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_38(c_void_p(buf141.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del primals_44
    buf147 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_187], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_200, buf146, reinterpret_tensor(primals_199, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf147)
    del primals_200
    buf148 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_39(c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    buf149 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_191], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_202, buf148, reinterpret_tensor(primals_201, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf149)
    del primals_202
    buf150 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_40(c_void_p(buf149.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf150.data_ptr()))
    # Source Nodes: [x_197], Original ATen: [aten.convolution]
    buf151 = extern_kernels.convolution(buf150, primals_203, primals_204, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf151, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_204
    buf152 = reinterpret_tensor(buf143, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf143  # reuse
    buf153 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf155 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf156 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_41(c_void_p(buf151.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    del primals_47
    buf157 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_201], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_206, buf156, reinterpret_tensor(primals_205, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf157)
    del primals_206
    buf158 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_42(c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_205], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_208, buf158, reinterpret_tensor(primals_207, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf159)
    del primals_208
    buf160 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_43(c_void_p(buf159.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf160.data_ptr()))
    # Source Nodes: [x_211], Original ATen: [aten.convolution]
    buf161 = extern_kernels.convolution(buf160, primals_209, primals_210, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf161, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_210
    buf162 = reinterpret_tensor(buf153, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf153  # reuse
    buf163 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf165 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf166 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_44(c_void_p(buf161.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    del primals_50
    buf167 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_215], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_212, buf166, reinterpret_tensor(primals_211, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf167)
    del primals_212
    buf168 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_45(c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_219], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_214, buf168, reinterpret_tensor(primals_213, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf169)
    del primals_214
    buf170 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_46(c_void_p(buf169.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf170.data_ptr()))
    # Source Nodes: [x_225], Original ATen: [aten.convolution]
    buf171 = extern_kernels.convolution(buf170, primals_215, primals_216, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf171, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_216
    buf172 = reinterpret_tensor(buf163, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf163  # reuse
    buf173 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf175 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf176 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_47(c_void_p(buf171.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    del primals_53
    buf177 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_229], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_218, buf176, reinterpret_tensor(primals_217, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf177)
    del primals_218
    buf178 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_48(c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    buf179 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_233], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_220, buf178, reinterpret_tensor(primals_219, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf179)
    del primals_220
    buf180 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_49(c_void_p(buf179.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf180.data_ptr()))
    # Source Nodes: [x_239], Original ATen: [aten.convolution]
    buf181 = extern_kernels.convolution(buf180, primals_221, primals_222, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf181, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_222
    buf182 = reinterpret_tensor(buf173, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf173  # reuse
    buf183 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf185 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf186 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_50(c_void_p(buf181.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del primals_56
    buf187 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_243], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_224, buf186, reinterpret_tensor(primals_223, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf187)
    del primals_224
    buf188 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_51(c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    buf189 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_247], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_226, buf188, reinterpret_tensor(primals_225, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf189)
    del primals_226
    buf190 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_52(c_void_p(buf189.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf190.data_ptr()))
    # Source Nodes: [x_253], Original ATen: [aten.convolution]
    buf191 = extern_kernels.convolution(buf190, primals_227, primals_228, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf191, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_228
    buf192 = reinterpret_tensor(buf183, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf183  # reuse
    buf193 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf195 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf196 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_53(c_void_p(buf191.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del primals_59
    buf197 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_257], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_230, buf196, reinterpret_tensor(primals_229, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf197)
    del primals_230
    buf198 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_54(c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    buf199 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_261], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_232, buf198, reinterpret_tensor(primals_231, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf199)
    del primals_232
    buf200 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_55(c_void_p(buf199.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf200.data_ptr()))
    # Source Nodes: [x_267], Original ATen: [aten.convolution]
    buf201 = extern_kernels.convolution(buf200, primals_233, primals_234, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf201, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_234
    buf202 = reinterpret_tensor(buf193, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf193  # reuse
    buf203 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf206 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_56(c_void_p(buf201.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del primals_62
    buf207 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_271], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_236, buf206, reinterpret_tensor(primals_235, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf207)
    del primals_236
    buf208 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_57(c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    buf209 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_275], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_238, buf208, reinterpret_tensor(primals_237, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf209)
    del primals_238
    buf210 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_58(c_void_p(buf209.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf210.data_ptr()))
    # Source Nodes: [x_281], Original ATen: [aten.convolution]
    buf211 = extern_kernels.convolution(buf210, primals_239, primals_240, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf211, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_240
    buf212 = reinterpret_tensor(buf203, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf203  # reuse
    buf213 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf215 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf216 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_59(c_void_p(buf211.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del primals_65
    buf217 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_285], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_242, buf216, reinterpret_tensor(primals_241, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf217)
    del primals_242
    buf218 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_60(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_289], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_244, buf218, reinterpret_tensor(primals_243, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf219)
    del primals_244
    buf220 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_61(c_void_p(buf219.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf220.data_ptr()))
    # Source Nodes: [x_295], Original ATen: [aten.convolution]
    buf221 = extern_kernels.convolution(buf220, primals_245, primals_246, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf221, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_246
    buf222 = reinterpret_tensor(buf213, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf213  # reuse
    buf223 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf226 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_62(c_void_p(buf221.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del primals_68
    buf227 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_299], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_248, buf226, reinterpret_tensor(primals_247, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf227)
    del primals_248
    buf228 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_63(c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    buf229 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_303], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_250, buf228, reinterpret_tensor(primals_249, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf229)
    del primals_250
    buf230 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_64(c_void_p(buf229.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf230.data_ptr()))
    # Source Nodes: [x_309], Original ATen: [aten.convolution]
    buf231 = extern_kernels.convolution(buf230, primals_251, primals_252, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf231, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_252
    buf232 = reinterpret_tensor(buf223, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf223  # reuse
    buf233 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf235 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf236 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_65(c_void_p(buf231.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    del primals_71
    buf237 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_313], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_254, buf236, reinterpret_tensor(primals_253, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf237)
    del primals_254
    buf238 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_66(c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_317], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_256, buf238, reinterpret_tensor(primals_255, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf239)
    del primals_256
    buf240 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_67(c_void_p(buf239.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf240.data_ptr()))
    # Source Nodes: [x_323], Original ATen: [aten.convolution]
    buf241 = extern_kernels.convolution(buf240, primals_257, primals_258, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf241, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_258
    buf242 = reinterpret_tensor(buf233, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf233  # reuse
    buf243 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf245 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf246 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_68(c_void_p(buf241.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    del primals_74
    buf247 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_327], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_260, buf246, reinterpret_tensor(primals_259, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf247)
    del primals_260
    buf248 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_69(c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    buf249 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_331], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_262, buf248, reinterpret_tensor(primals_261, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf249)
    del primals_262
    buf250 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_70(c_void_p(buf249.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf250.data_ptr()))
    # Source Nodes: [x_337], Original ATen: [aten.convolution]
    buf251 = extern_kernels.convolution(buf250, primals_263, primals_264, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf251, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_264
    buf252 = reinterpret_tensor(buf243, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf243  # reuse
    buf253 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf255 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf256 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_71(c_void_p(buf251.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del primals_77
    buf257 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_341], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_266, buf256, reinterpret_tensor(primals_265, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf257)
    del primals_266
    buf258 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_72(c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    buf259 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_345], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_268, buf258, reinterpret_tensor(primals_267, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf259)
    del primals_268
    buf260 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_73(c_void_p(buf259.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf260.data_ptr()))
    # Source Nodes: [x_351], Original ATen: [aten.convolution]
    buf261 = extern_kernels.convolution(buf260, primals_269, primals_270, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf261, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_270
    buf262 = reinterpret_tensor(buf253, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf253  # reuse
    buf263 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf265 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf266 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_74(c_void_p(buf261.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del primals_80
    buf267 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_355], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_272, buf266, reinterpret_tensor(primals_271, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf267)
    del primals_272
    buf268 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_75(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_359], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_274, buf268, reinterpret_tensor(primals_273, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf269)
    del primals_274
    buf270 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_76(c_void_p(buf269.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf270.data_ptr()))
    # Source Nodes: [x_365], Original ATen: [aten.convolution]
    buf271 = extern_kernels.convolution(buf270, primals_275, primals_276, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf271, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_276
    buf272 = reinterpret_tensor(buf263, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf263  # reuse
    buf273 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf275 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf276 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_77(c_void_p(buf271.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del primals_83
    buf277 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_369], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_278, buf276, reinterpret_tensor(primals_277, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf277)
    del primals_278
    buf278 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_78(c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    buf279 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_373], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_280, buf278, reinterpret_tensor(primals_279, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf279)
    del primals_280
    buf280 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_79(c_void_p(buf279.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf280.data_ptr()))
    # Source Nodes: [x_379], Original ATen: [aten.convolution]
    buf281 = extern_kernels.convolution(buf280, primals_281, primals_282, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf281, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_282
    buf282 = reinterpret_tensor(buf273, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf273  # reuse
    buf283 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf285 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf286 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_80(c_void_p(buf281.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del primals_86
    buf287 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_383], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_284, buf286, reinterpret_tensor(primals_283, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf287)
    del primals_284
    buf288 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_81(c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    buf289 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_387], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_286, buf288, reinterpret_tensor(primals_285, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf289)
    del primals_286
    buf290 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_82(c_void_p(buf289.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf290.data_ptr()))
    # Source Nodes: [x_393], Original ATen: [aten.convolution]
    buf291 = extern_kernels.convolution(buf290, primals_287, primals_288, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf291, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_288
    buf292 = reinterpret_tensor(buf283, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf283  # reuse
    buf293 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf295 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf296 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_83(c_void_p(buf291.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    del primals_89
    buf297 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_397], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_290, buf296, reinterpret_tensor(primals_289, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf297)
    del primals_290
    buf298 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_84(c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    buf299 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_401], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_292, buf298, reinterpret_tensor(primals_291, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf299)
    del primals_292
    buf300 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_85(c_void_p(buf299.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf300.data_ptr()))
    # Source Nodes: [x_407], Original ATen: [aten.convolution]
    buf301 = extern_kernels.convolution(buf300, primals_293, primals_294, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf301, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_294
    buf302 = reinterpret_tensor(buf293, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf293  # reuse
    buf303 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf305 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf306 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_86(c_void_p(buf301.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del primals_92
    buf307 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_411], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_296, buf306, reinterpret_tensor(primals_295, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf307)
    del primals_296
    buf308 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_87(c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    buf309 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_415], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_298, buf308, reinterpret_tensor(primals_297, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf309)
    del primals_298
    buf310 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_88(c_void_p(buf309.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf310.data_ptr()))
    # Source Nodes: [x_421], Original ATen: [aten.convolution]
    buf311 = extern_kernels.convolution(buf310, primals_299, primals_300, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf311, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_300
    buf312 = reinterpret_tensor(buf303, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf303  # reuse
    buf313 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf315 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf316 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_89(c_void_p(buf311.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()))
    del primals_95
    buf317 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_425], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_302, buf316, reinterpret_tensor(primals_301, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf317)
    del primals_302
    buf318 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_90(c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    buf319 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_429], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_304, buf318, reinterpret_tensor(primals_303, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf319)
    del primals_304
    buf320 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_91(c_void_p(buf319.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf320.data_ptr()))
    # Source Nodes: [x_435], Original ATen: [aten.convolution]
    buf321 = extern_kernels.convolution(buf320, primals_305, primals_306, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf321, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_306
    buf322 = reinterpret_tensor(buf313, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf313  # reuse
    buf323 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf325 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf326 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_92(c_void_p(buf321.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del primals_98
    buf327 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_439], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_308, buf326, reinterpret_tensor(primals_307, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf327)
    del primals_308
    buf328 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_93(c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    buf329 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_443], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_310, buf328, reinterpret_tensor(primals_309, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf329)
    del primals_310
    buf330 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_94(c_void_p(buf329.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf330.data_ptr()))
    # Source Nodes: [x_449], Original ATen: [aten.convolution]
    buf331 = extern_kernels.convolution(buf330, primals_311, primals_312, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf331, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_312
    buf332 = reinterpret_tensor(buf323, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf323  # reuse
    buf333 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf335 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf336 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_95(c_void_p(buf331.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    del primals_101
    buf337 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_453], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_314, buf336, reinterpret_tensor(primals_313, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf337)
    del primals_314
    buf338 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_96(c_void_p(buf337.data_ptr()), c_void_p(buf338.data_ptr()))
    buf339 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_457], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_316, buf338, reinterpret_tensor(primals_315, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf339)
    del primals_316
    buf340 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_97(c_void_p(buf339.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf340.data_ptr()))
    # Source Nodes: [x_463], Original ATen: [aten.convolution]
    buf341 = extern_kernels.convolution(buf340, primals_317, primals_318, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf341, (8, 512, 14, 14), (100352, 1, 7168, 512))
    del primals_318
    buf342 = reinterpret_tensor(buf333, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf333  # reuse
    buf343 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf345 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf346 = empty((1568, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_98(c_void_p(buf341.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    del primals_104
    buf347 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_467], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_320, buf346, reinterpret_tensor(primals_319, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf347)
    del primals_320
    buf348 = empty((1568, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_99(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    buf349 = empty((1568, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_471], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_322, buf348, reinterpret_tensor(primals_321, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf349)
    del primals_322
    buf350 = buf343; del buf343  # reuse
    buf351 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf353 = empty_strided((8, 14, 14, 512), (100352, 1, 7168, 14), device='cpu', dtype=torch.float32)
    buf354 = empty_strided((8, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_permute_100(c_void_p(buf349.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    del primals_107
    # Source Nodes: [shortcut_33], Original ATen: [aten.convolution]
    buf355 = extern_kernels.convolution(buf354, buf3, primals_324, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf355, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    del primals_324
    # Source Nodes: [x_482], Original ATen: [aten.convolution]
    buf356 = extern_kernels.convolution(buf355, primals_325, primals_326, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024)
    assert_size_stride(buf356, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    del primals_326
    buf357 = empty_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    buf358 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cpu', dtype=torch.float32)
    buf360 = empty_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    buf361 = empty((392, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_101(c_void_p(buf356.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del primals_109
    buf362 = empty((392, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_486], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_328, buf361, reinterpret_tensor(primals_327, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf362)
    del primals_328
    buf363 = empty((392, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_102(c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_490], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_330, buf363, reinterpret_tensor(primals_329, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf364)
    del primals_330
    buf365 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_103(c_void_p(buf364.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf365.data_ptr()))
    # Source Nodes: [x_496], Original ATen: [aten.convolution]
    buf366 = extern_kernels.convolution(buf365, primals_331, primals_332, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024)
    assert_size_stride(buf366, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    del primals_332
    buf367 = reinterpret_tensor(buf358, (8, 7, 7, 1), (49, 1, 7, 7), 0); del buf358  # reuse
    buf368 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cpu', dtype=torch.float32)
    buf370 = empty_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    buf371 = empty((392, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_104(c_void_p(buf366.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    del primals_112
    buf372 = empty((392, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_500], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_334, buf371, reinterpret_tensor(primals_333, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf372)
    del primals_334
    buf373 = empty((392, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_105(c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()))
    buf374 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_504], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_336, buf373, reinterpret_tensor(primals_335, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf374)
    del primals_336
    buf375 = empty_strided((8, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_106(c_void_p(buf374.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf375.data_ptr()))
    # Source Nodes: [x_510], Original ATen: [aten.convolution]
    buf376 = extern_kernels.convolution(buf375, primals_337, primals_338, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024)
    assert_size_stride(buf376, (8, 1024, 7, 7), (50176, 1, 7168, 1024))
    del primals_338
    buf377 = reinterpret_tensor(buf368, (8, 7, 7, 1), (49, 1, 7, 7), 0); del buf368  # reuse
    buf378 = empty_strided((8, 7, 7, 1), (49, 7, 1, 392), device='cpu', dtype=torch.float32)
    buf380 = empty_strided((8, 7, 7, 1), (49, 1, 7, 7), device='cpu', dtype=torch.float32)
    buf381 = empty((392, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_107(c_void_p(buf376.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    del buf378
    del primals_115
    buf382 = empty((392, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_514], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_340, buf381, reinterpret_tensor(primals_339, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf382)
    del primals_340
    buf383 = empty((392, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_108(c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    buf384 = empty((392, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_518], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_342, buf383, reinterpret_tensor(primals_341, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf384)
    del primals_342
    buf385 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cpu', dtype=torch.float32)
    buf386 = reinterpret_tensor(buf385, (8, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf385  # reuse
    buf387 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cpu', dtype=torch.float32)
    buf388 = empty_strided((8, 1, 1, 1), (1, 8, 8, 8), device='cpu', dtype=torch.float32)
    buf390 = empty_strided((8, 1, 1, 1024), (1024, 1, 1024, 1), device='cpu', dtype=torch.float32)
    buf391 = empty((8, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_mean_mul_native_layer_norm_view_109(c_void_p(buf386.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    del buf386
    del buf387
    del primals_118
    buf392 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_344, buf391, reinterpret_tensor(primals_343, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf392)
    del primals_344
    buf393 = reinterpret_tensor(buf388, (8, 1, 1, 1), (1, 1, 1, 1), 0); del buf388  # reuse
    buf394 = reinterpret_tensor(buf350, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf350  # reuse
    buf395 = reinterpret_tensor(buf75, (8, 28, 28, 1), (784, 1, 28, 28), 0); del buf75  # reuse
    buf396 = reinterpret_tensor(buf40, (8, 56, 56, 1), (3136, 1, 56, 56), 0); del buf40  # reuse
    buf397 = empty_strided((8, 56, 56, 1), (3136, 1, 56, 56), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_native_layer_norm_backward_110(c_void_p(buf393.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()))
    return (buf392, primals_1, primals_3, primals_5, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_110, primals_111, primals_113, primals_114, primals_116, primals_117, buf0, primals_121, primals_127, primals_133, buf1, primals_141, primals_147, primals_153, buf2, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, buf3, primals_325, primals_331, primals_337, buf4, buf9, buf10, buf11, buf12, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf35, buf36, buf37, buf38, buf39, buf43, buf44, buf45, buf46, buf47, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf70, buf71, buf72, buf73, buf74, buf78, buf79, buf80, buf81, buf82, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf222, buf225, buf226, buf227, buf228, buf229, buf230, buf231, buf232, buf235, buf236, buf237, buf238, buf239, buf240, buf241, buf242, buf245, buf246, buf247, buf248, buf249, buf250, buf251, buf252, buf255, buf256, buf257, buf258, buf259, buf260, buf261, buf262, buf265, buf266, buf267, buf268, buf269, buf270, buf271, buf272, buf275, buf276, buf277, buf278, buf279, buf280, buf281, buf282, buf285, buf286, buf287, buf288, buf289, buf290, buf291, buf292, buf295, buf296, buf297, buf298, buf299, buf300, buf301, buf302, buf305, buf306, buf307, buf308, buf309, buf310, buf311, buf312, buf315, buf316, buf317, buf318, buf319, buf320, buf321, buf322, buf325, buf326, buf327, buf328, buf329, buf330, buf331, buf332, buf335, buf336, buf337, buf338, buf339, buf340, buf341, buf342, buf345, buf346, buf347, buf348, buf349, buf353, buf354, buf355, buf356, buf357, buf360, buf361, buf362, buf363, buf364, buf365, buf366, buf367, buf370, buf371, buf372, buf373, buf374, buf375, buf376, buf377, buf380, buf381, buf382, buf383, buf384, buf390, buf391, reinterpret_tensor(primals_343, (1000, 1024), (1024, 1), 0), buf393, reinterpret_tensor(primals_341, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_339, (4096, 1024), (1024, 1), 0), reinterpret_tensor(primals_335, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_333, (4096, 1024), (1024, 1), 0), reinterpret_tensor(primals_329, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_327, (4096, 1024), (1024, 1), 0), buf394, reinterpret_tensor(primals_321, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_319, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_315, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_313, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_309, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_307, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_303, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_301, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_297, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_295, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_291, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_289, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_285, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_283, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_279, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_277, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_273, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_271, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_267, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_265, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_261, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_259, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_255, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_253, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_249, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_247, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_243, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_241, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_237, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_235, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_231, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_229, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_225, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_223, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_219, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_217, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_213, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_211, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_207, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_205, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_201, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_199, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_195, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_193, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_189, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_187, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_183, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_181, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_177, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_175, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_171, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_169, (2048, 512), (512, 1), 0), reinterpret_tensor(primals_165, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_163, (2048, 512), (512, 1), 0), buf395, reinterpret_tensor(primals_157, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_155, (1024, 256), (256, 1), 0), reinterpret_tensor(primals_151, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_149, (1024, 256), (256, 1), 0), reinterpret_tensor(primals_145, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_143, (1024, 256), (256, 1), 0), buf396, reinterpret_tensor(primals_137, (128, 512), (512, 1), 0), reinterpret_tensor(primals_135, (512, 128), (128, 1), 0), reinterpret_tensor(primals_131, (128, 512), (512, 1), 0), reinterpret_tensor(primals_129, (512, 128), (128, 1), 0), reinterpret_tensor(primals_125, (128, 512), (512, 1), 0), reinterpret_tensor(primals_123, (512, 128), (128, 1), 0), buf397, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
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
    primals_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((128, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((128, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((512, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((128, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((256, 128, 2, 2), (512, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((256, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((512, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((1024, 512, 2, 2), (2048, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((1024, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convnext_base', benchmark_compiled_module)
