
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_1 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((24L*(static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(14L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x2, 96L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((x1 + x1_inner), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x2, 96L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((x1 + x1_inner), 14L))) + (75264L*x0) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x2, 96L))) + (16L*(static_cast<long>(x2) % static_cast<long>(24L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = in_ptr1[static_cast<long>((16L*(static_cast<long>(x2) % static_cast<long>(24L))) + (c10::div_floor_integer(x2, 24L)))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 + tmp2;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>((24L*(static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>(x1) % static_cast<long>(14L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>(x1) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x2, 96L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x2, 96L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer(x1, 14L))) + (75264L*x0) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x2, 96L))) + (16L*(static_cast<long>(x2) % static_cast<long>(24L))) + (static_cast<long>(c10::div_floor_integer(x2, 24L)) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))];
                            auto tmp1 = in_ptr1[static_cast<long>((16L*(static_cast<long>(x2) % static_cast<long>(24L))) + (c10::div_floor_integer(x2, 24L)))];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp2);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((24L*(static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 24L)) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>(x1) % static_cast<long>(14L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 24L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>(x1) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x2 + x2_inner), 96L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 24L)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer(x1, 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x2 + x2_inner), 96L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 24L)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer(x1, 14L))) + (75264L*x0) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x2 + x2_inner), 96L))) + (16L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(24L))) + (static_cast<long>(c10::div_floor_integer((x2 + x2_inner), 24L)) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((16L*(static_cast<long>((x2 + x2_inner)) % static_cast<long>(24L))) + (c10::div_floor_integer((x2 + x2_inner), 24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(384.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_2 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>((24L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (16L*x2) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x2)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.mean.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>((24L*(static_cast<long>(x1) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (static_cast<long>(x1) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (16L*x2) + (16L*x2_inner) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>(x1 + (16L*x2) + (16L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = out_ptr2[static_cast<long>(x1 + (16L*x0))];
                        auto tmp6 = out_ptr3[static_cast<long>(x1 + (16L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(24.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr4 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((24L*(static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer((x1 + x1_inner), 4L))) + (16L*x2) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2)));
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (24L*x1) + (24L*x1_inner) + (384L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((24L*(static_cast<long>(x1) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (static_cast<long>(x1) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (16L*x2) + (16L*x2_inner) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (16L*x2) + (16L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(24.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        tmp19.store(out_ptr2 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((24L*(static_cast<long>(x1) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (static_cast<long>(x1) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (16L*x2) + (16L*x2_inner) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (16L*x2) + (16L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                        out_ptr2[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr3[static_cast<long>(x1 + (16L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (384L*x1)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr5 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp13 = in_ptr6[static_cast<long>((-1L) + x1 + (196L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp16 = in_ptr7[static_cast<long>((-1L) + x1 + (196L*x0))];
                            auto tmp17 = static_cast<float>(384.0);
                            auto tmp18 = tmp16 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = 1 / std::sqrt(tmp20);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp15 * tmp22;
                            auto tmp24 = masked_load(in_ptr8 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp25 = tmp23 * tmp24;
                            auto tmp26 = masked_load(in_ptr9 + static_cast<long>(x2), to_float_mask(tmp8));
                            auto tmp27 = tmp25 + tmp26;
                            return tmp27;
                        }
                        ;
                        auto tmp28 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp29 = to_float_mask(tmp4);
                        auto tmp30 = decltype(tmp7)::blendv(tmp28, tmp7, tmp29);
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((24L*(static_cast<long>(x1) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (static_cast<long>(x1) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (16L*x2) + (16L*x2_inner) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (16L*x2) + (16L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (16L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (16L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x2));
                        auto tmp22 = out_ptr2[static_cast<long>(x1 + (16L*x0))];
                        auto tmp25 = out_ptr3[static_cast<long>(x1 + (16L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(24.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        auto tmp23 = at::vec::Vectorized<float>(tmp22);
                        auto tmp24 = tmp6 - tmp23;
                        auto tmp26 = tmp25 / tmp11;
                        auto tmp27 = decltype(tmp26)(tmp26 + tmp13);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp24 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp21.store(out_ptr5 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        tmp34.store(out_ptr6 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_10 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((24L*(static_cast<long>(x1) % static_cast<long>(4L))) + (24L*((((4L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (static_cast<long>(x1) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (96L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(196L))) % static_cast<long>(14L))) + (1344L*(static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) + (1344L*((((4L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (static_cast<long>(x1) % static_cast<long>(4L))), 4L)) % static_cast<long>(4L))) >= 0L) ? 0L : 56L)) + (5376L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(196L)), 14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(((4L*(c10::div_floor_integer(x1, 4L))) + (16L*x2) + (16L*x2_inner) + (static_cast<long>(x1) % static_cast<long>(4L))), 16L)) % static_cast<long>(24L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr1[static_cast<long>(x1 + (16L*x2) + (16L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (24L*x1) + (384L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_22 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr2 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr3 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = in_ptr5[static_cast<long>(x0)];
                    auto tmp6 = in_ptr6[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = in_ptr9[static_cast<long>(x0)];
                    auto tmp21 = in_ptr10[static_cast<long>(x0)];
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp2 - tmp19;
                    auto tmp22 = tmp21 / tmp7;
                    auto tmp23 = decltype(tmp22)(tmp22 + tmp9);
                    auto tmp24 = 1 / std::sqrt(tmp23);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp20 * tmp25;
                    auto tmp28 = tmp26 * tmp27;
                    auto tmp30 = tmp28 + tmp29;
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_25 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_29 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_37 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr7[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr8[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr5 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = in_ptr7[static_cast<long>(x0)];
                    auto tmp10 = in_ptr8[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp22 = in_ptr11[static_cast<long>(x0)];
                    auto tmp25 = in_ptr12[static_cast<long>(x0)];
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = at::vec::Vectorized<float>(tmp22);
                    auto tmp24 = tmp6 - tmp23;
                    auto tmp26 = tmp25 / tmp11;
                    auto tmp27 = decltype(tmp26)(tmp26 + tmp13);
                    auto tmp28 = 1 / std::sqrt(tmp27);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp24 * tmp29;
                    auto tmp32 = tmp30 * tmp31;
                    auto tmp34 = tmp32 + tmp33;
                    tmp21.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp34.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_40 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_44 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_52 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr2 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr3 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = in_ptr5[static_cast<long>(x0)];
                    auto tmp6 = in_ptr6[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = in_ptr9[static_cast<long>(x0)];
                    auto tmp21 = in_ptr10[static_cast<long>(x0)];
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp2 - tmp19;
                    auto tmp22 = tmp21 / tmp7;
                    auto tmp23 = decltype(tmp22)(tmp22 + tmp9);
                    auto tmp24 = 1 / std::sqrt(tmp23);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp20 * tmp25;
                    auto tmp28 = tmp26 * tmp27;
                    auto tmp30 = tmp28 + tmp29;
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_55 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_56 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_59 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_64 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_67 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr7[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr8[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr5 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = in_ptr7[static_cast<long>(x0)];
                    auto tmp10 = in_ptr8[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp22 = in_ptr11[static_cast<long>(x0)];
                    auto tmp25 = in_ptr12[static_cast<long>(x0)];
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = at::vec::Vectorized<float>(tmp22);
                    auto tmp24 = tmp6 - tmp23;
                    auto tmp26 = tmp25 / tmp11;
                    auto tmp27 = decltype(tmp26)(tmp26 + tmp13);
                    auto tmp28 = 1 / std::sqrt(tmp27);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp24 * tmp29;
                    auto tmp32 = tmp30 * tmp31;
                    auto tmp34 = tmp32 + tmp33;
                    tmp21.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp34.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_70 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_71 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_74 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_82 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr2 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr3 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = in_ptr5[static_cast<long>(x0)];
                    auto tmp6 = in_ptr6[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = in_ptr9[static_cast<long>(x0)];
                    auto tmp21 = in_ptr10[static_cast<long>(x0)];
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp2 - tmp19;
                    auto tmp22 = tmp21 / tmp7;
                    auto tmp23 = decltype(tmp22)(tmp22 + tmp9);
                    auto tmp24 = 1 / std::sqrt(tmp23);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp20 * tmp25;
                    auto tmp28 = tmp26 * tmp27;
                    auto tmp30 = tmp28 + tmp29;
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_85 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_86 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_89 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_94 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_97 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr7[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr8[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr5 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = in_ptr7[static_cast<long>(x0)];
                    auto tmp10 = in_ptr8[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp22 = in_ptr11[static_cast<long>(x0)];
                    auto tmp25 = in_ptr12[static_cast<long>(x0)];
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = at::vec::Vectorized<float>(tmp22);
                    auto tmp24 = tmp6 - tmp23;
                    auto tmp26 = tmp25 / tmp11;
                    auto tmp27 = decltype(tmp26)(tmp26 + tmp13);
                    auto tmp28 = 1 / std::sqrt(tmp27);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp24 * tmp29;
                    auto tmp32 = tmp30 * tmp31;
                    auto tmp34 = tmp32 + tmp33;
                    tmp21.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp34.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_100 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_101 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_104 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_105 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_109 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_110 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_112 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr2 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr3 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = in_ptr5[static_cast<long>(x0)];
                    auto tmp6 = in_ptr6[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = in_ptr9[static_cast<long>(x0)];
                    auto tmp21 = in_ptr10[static_cast<long>(x0)];
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp2 - tmp19;
                    auto tmp22 = tmp21 / tmp7;
                    auto tmp23 = decltype(tmp22)(tmp22 + tmp9);
                    auto tmp24 = 1 / std::sqrt(tmp23);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp20 * tmp25;
                    auto tmp28 = tmp26 * tmp27;
                    auto tmp30 = tmp28 + tmp29;
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_115 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_116 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_119 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_120 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_124 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_125 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_127 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr7[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr8[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr5 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = in_ptr7[static_cast<long>(x0)];
                    auto tmp10 = in_ptr8[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp22 = in_ptr11[static_cast<long>(x0)];
                    auto tmp25 = in_ptr12[static_cast<long>(x0)];
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = at::vec::Vectorized<float>(tmp22);
                    auto tmp24 = tmp6 - tmp23;
                    auto tmp26 = tmp25 / tmp11;
                    auto tmp27 = decltype(tmp26)(tmp26 + tmp13);
                    auto tmp28 = 1 / std::sqrt(tmp27);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp24 * tmp29;
                    auto tmp32 = tmp30 * tmp31;
                    auto tmp34 = tmp32 + tmp33;
                    tmp21.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp34.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_130 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_131 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_134 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_135 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_136 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_138 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_140 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_142 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr2 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr3 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_144 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = in_ptr5[static_cast<long>(x0)];
                    auto tmp6 = in_ptr6[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp18 = in_ptr9[static_cast<long>(x0)];
                    auto tmp21 = in_ptr10[static_cast<long>(x0)];
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp2 - tmp19;
                    auto tmp22 = tmp21 / tmp7;
                    auto tmp23 = decltype(tmp22)(tmp22 + tmp9);
                    auto tmp24 = 1 / std::sqrt(tmp23);
                    auto tmp25 = at::vec::Vectorized<float>(tmp24);
                    auto tmp26 = tmp20 * tmp25;
                    auto tmp28 = tmp26 * tmp27;
                    auto tmp30 = tmp28 + tmp29;
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp30.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_145 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_146 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_147 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_148 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_149 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_150 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_151 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_153 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_154 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_155 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(24.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_156 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_157 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr7[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr8[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr4 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr5 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_159 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp7 = in_ptr7[static_cast<long>(x0)];
                    auto tmp10 = in_ptr8[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp22 = in_ptr11[static_cast<long>(x0)];
                    auto tmp25 = in_ptr12[static_cast<long>(x0)];
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                    auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(24.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    auto tmp23 = at::vec::Vectorized<float>(tmp22);
                    auto tmp24 = tmp6 - tmp23;
                    auto tmp26 = tmp25 / tmp11;
                    auto tmp27 = decltype(tmp26)(tmp26 + tmp13);
                    auto tmp28 = 1 / std::sqrt(tmp27);
                    auto tmp29 = at::vec::Vectorized<float>(tmp28);
                    auto tmp30 = tmp24 * tmp29;
                    auto tmp32 = tmp30 * tmp31;
                    auto tmp34 = tmp32 + tmp33;
                    tmp21.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    tmp34.store(out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_160 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_161 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_162 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_163 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_164 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_165 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_166 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (48L*x2) + (768L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(24L + x1 + (48L*x2) + (768L*x0)), static_cast<long>(48L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (384L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.408248290463863);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_168 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x3 + (6L*x1) + (24L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (96L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_169 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(6L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (6L*x1) + (96L*x2) + (384L*x0))];
                            out_ptr0[static_cast<long>(x3 + (6L*x2) + (24L*x1) + (384L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_170 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(24.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_171 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_172 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = in_ptr5[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = in_ptr6[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr2 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr3 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_173 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_174 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = in_ptr5[static_cast<long>(x0)];
                    auto tmp6 = in_ptr6[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(24.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_175 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_176 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_177 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_178 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9456L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (384L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (12608L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_179 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(6L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (12608L*x2) + (75648L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (384L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_180 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(384.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-05);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_181 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2420736L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_clone_native_layer_norm_182 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>((-384L) + x2 + (384L*x1) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (75648L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(197L*x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(197L*x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(384.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 24, 4, 4), (384, 16, 4, 1))
    assert_size_stride(arg1_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg2_1, (1, 197, 384), (75648, 384, 1))
    assert_size_stride(arg3_1, (24, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg4_1, (24, ), (1, ))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (384, ), (1, ))
    assert_size_stride(arg7_1, (384, 384), (384, 1))
    assert_size_stride(arg8_1, (384, ), (1, ))
    assert_size_stride(arg9_1, (384, ), (1, ))
    assert_size_stride(arg10_1, (384, ), (1, ))
    assert_size_stride(arg11_1, (24, ), (1, ))
    assert_size_stride(arg12_1, (24, ), (1, ))
    assert_size_stride(arg13_1, (48, 24), (24, 1))
    assert_size_stride(arg14_1, (24, 24), (24, 1))
    assert_size_stride(arg15_1, (24, 24), (24, 1))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (24, ), (1, ))
    assert_size_stride(arg19_1, (96, 24), (24, 1))
    assert_size_stride(arg20_1, (96, ), (1, ))
    assert_size_stride(arg21_1, (24, 96), (96, 1))
    assert_size_stride(arg22_1, (24, ), (1, ))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (24, ), (1, ))
    assert_size_stride(arg25_1, (384, 384), (384, 1))
    assert_size_stride(arg26_1, (384, ), (1, ))
    assert_size_stride(arg27_1, (384, ), (1, ))
    assert_size_stride(arg28_1, (384, ), (1, ))
    assert_size_stride(arg29_1, (768, 384), (384, 1))
    assert_size_stride(arg30_1, (384, 384), (384, 1))
    assert_size_stride(arg31_1, (384, 384), (384, 1))
    assert_size_stride(arg32_1, (384, ), (1, ))
    assert_size_stride(arg33_1, (384, ), (1, ))
    assert_size_stride(arg34_1, (384, ), (1, ))
    assert_size_stride(arg35_1, (1536, 384), (384, 1))
    assert_size_stride(arg36_1, (1536, ), (1, ))
    assert_size_stride(arg37_1, (384, 1536), (1536, 1))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (24, ), (1, ))
    assert_size_stride(arg40_1, (24, ), (1, ))
    assert_size_stride(arg41_1, (48, 24), (24, 1))
    assert_size_stride(arg42_1, (24, 24), (24, 1))
    assert_size_stride(arg43_1, (24, 24), (24, 1))
    assert_size_stride(arg44_1, (24, ), (1, ))
    assert_size_stride(arg45_1, (24, ), (1, ))
    assert_size_stride(arg46_1, (24, ), (1, ))
    assert_size_stride(arg47_1, (96, 24), (24, 1))
    assert_size_stride(arg48_1, (96, ), (1, ))
    assert_size_stride(arg49_1, (24, 96), (96, 1))
    assert_size_stride(arg50_1, (24, ), (1, ))
    assert_size_stride(arg51_1, (24, ), (1, ))
    assert_size_stride(arg52_1, (24, ), (1, ))
    assert_size_stride(arg53_1, (384, 384), (384, 1))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (384, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (768, 384), (384, 1))
    assert_size_stride(arg58_1, (384, 384), (384, 1))
    assert_size_stride(arg59_1, (384, 384), (384, 1))
    assert_size_stride(arg60_1, (384, ), (1, ))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (1536, 384), (384, 1))
    assert_size_stride(arg64_1, (1536, ), (1, ))
    assert_size_stride(arg65_1, (384, 1536), (1536, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (24, ), (1, ))
    assert_size_stride(arg68_1, (24, ), (1, ))
    assert_size_stride(arg69_1, (48, 24), (24, 1))
    assert_size_stride(arg70_1, (24, 24), (24, 1))
    assert_size_stride(arg71_1, (24, 24), (24, 1))
    assert_size_stride(arg72_1, (24, ), (1, ))
    assert_size_stride(arg73_1, (24, ), (1, ))
    assert_size_stride(arg74_1, (24, ), (1, ))
    assert_size_stride(arg75_1, (96, 24), (24, 1))
    assert_size_stride(arg76_1, (96, ), (1, ))
    assert_size_stride(arg77_1, (24, 96), (96, 1))
    assert_size_stride(arg78_1, (24, ), (1, ))
    assert_size_stride(arg79_1, (24, ), (1, ))
    assert_size_stride(arg80_1, (24, ), (1, ))
    assert_size_stride(arg81_1, (384, 384), (384, 1))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (768, 384), (384, 1))
    assert_size_stride(arg86_1, (384, 384), (384, 1))
    assert_size_stride(arg87_1, (384, 384), (384, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (1536, 384), (384, 1))
    assert_size_stride(arg92_1, (1536, ), (1, ))
    assert_size_stride(arg93_1, (384, 1536), (1536, 1))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (24, ), (1, ))
    assert_size_stride(arg96_1, (24, ), (1, ))
    assert_size_stride(arg97_1, (48, 24), (24, 1))
    assert_size_stride(arg98_1, (24, 24), (24, 1))
    assert_size_stride(arg99_1, (24, 24), (24, 1))
    assert_size_stride(arg100_1, (24, ), (1, ))
    assert_size_stride(arg101_1, (24, ), (1, ))
    assert_size_stride(arg102_1, (24, ), (1, ))
    assert_size_stride(arg103_1, (96, 24), (24, 1))
    assert_size_stride(arg104_1, (96, ), (1, ))
    assert_size_stride(arg105_1, (24, 96), (96, 1))
    assert_size_stride(arg106_1, (24, ), (1, ))
    assert_size_stride(arg107_1, (24, ), (1, ))
    assert_size_stride(arg108_1, (24, ), (1, ))
    assert_size_stride(arg109_1, (384, 384), (384, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (768, 384), (384, 1))
    assert_size_stride(arg114_1, (384, 384), (384, 1))
    assert_size_stride(arg115_1, (384, 384), (384, 1))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (384, ), (1, ))
    assert_size_stride(arg119_1, (1536, 384), (384, 1))
    assert_size_stride(arg120_1, (1536, ), (1, ))
    assert_size_stride(arg121_1, (384, 1536), (1536, 1))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (24, ), (1, ))
    assert_size_stride(arg124_1, (24, ), (1, ))
    assert_size_stride(arg125_1, (48, 24), (24, 1))
    assert_size_stride(arg126_1, (24, 24), (24, 1))
    assert_size_stride(arg127_1, (24, 24), (24, 1))
    assert_size_stride(arg128_1, (24, ), (1, ))
    assert_size_stride(arg129_1, (24, ), (1, ))
    assert_size_stride(arg130_1, (24, ), (1, ))
    assert_size_stride(arg131_1, (96, 24), (24, 1))
    assert_size_stride(arg132_1, (96, ), (1, ))
    assert_size_stride(arg133_1, (24, 96), (96, 1))
    assert_size_stride(arg134_1, (24, ), (1, ))
    assert_size_stride(arg135_1, (24, ), (1, ))
    assert_size_stride(arg136_1, (24, ), (1, ))
    assert_size_stride(arg137_1, (384, 384), (384, 1))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (768, 384), (384, 1))
    assert_size_stride(arg142_1, (384, 384), (384, 1))
    assert_size_stride(arg143_1, (384, 384), (384, 1))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (1536, 384), (384, 1))
    assert_size_stride(arg148_1, (1536, ), (1, ))
    assert_size_stride(arg149_1, (384, 1536), (1536, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (24, ), (1, ))
    assert_size_stride(arg152_1, (24, ), (1, ))
    assert_size_stride(arg153_1, (48, 24), (24, 1))
    assert_size_stride(arg154_1, (24, 24), (24, 1))
    assert_size_stride(arg155_1, (24, 24), (24, 1))
    assert_size_stride(arg156_1, (24, ), (1, ))
    assert_size_stride(arg157_1, (24, ), (1, ))
    assert_size_stride(arg158_1, (24, ), (1, ))
    assert_size_stride(arg159_1, (96, 24), (24, 1))
    assert_size_stride(arg160_1, (96, ), (1, ))
    assert_size_stride(arg161_1, (24, 96), (96, 1))
    assert_size_stride(arg162_1, (24, ), (1, ))
    assert_size_stride(arg163_1, (24, ), (1, ))
    assert_size_stride(arg164_1, (24, ), (1, ))
    assert_size_stride(arg165_1, (384, 384), (384, 1))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (768, 384), (384, 1))
    assert_size_stride(arg170_1, (384, 384), (384, 1))
    assert_size_stride(arg171_1, (384, 384), (384, 1))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (1536, 384), (384, 1))
    assert_size_stride(arg176_1, (1536, ), (1, ))
    assert_size_stride(arg177_1, (384, 1536), (1536, 1))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (24, ), (1, ))
    assert_size_stride(arg180_1, (24, ), (1, ))
    assert_size_stride(arg181_1, (48, 24), (24, 1))
    assert_size_stride(arg182_1, (24, 24), (24, 1))
    assert_size_stride(arg183_1, (24, 24), (24, 1))
    assert_size_stride(arg184_1, (24, ), (1, ))
    assert_size_stride(arg185_1, (24, ), (1, ))
    assert_size_stride(arg186_1, (24, ), (1, ))
    assert_size_stride(arg187_1, (96, 24), (24, 1))
    assert_size_stride(arg188_1, (96, ), (1, ))
    assert_size_stride(arg189_1, (24, 96), (96, 1))
    assert_size_stride(arg190_1, (24, ), (1, ))
    assert_size_stride(arg191_1, (24, ), (1, ))
    assert_size_stride(arg192_1, (24, ), (1, ))
    assert_size_stride(arg193_1, (384, 384), (384, 1))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (384, ), (1, ))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (768, 384), (384, 1))
    assert_size_stride(arg198_1, (384, 384), (384, 1))
    assert_size_stride(arg199_1, (384, 384), (384, 1))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (384, ), (1, ))
    assert_size_stride(arg203_1, (1536, 384), (384, 1))
    assert_size_stride(arg204_1, (1536, ), (1, ))
    assert_size_stride(arg205_1, (384, 1536), (1536, 1))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (24, ), (1, ))
    assert_size_stride(arg208_1, (24, ), (1, ))
    assert_size_stride(arg209_1, (48, 24), (24, 1))
    assert_size_stride(arg210_1, (24, 24), (24, 1))
    assert_size_stride(arg211_1, (24, 24), (24, 1))
    assert_size_stride(arg212_1, (24, ), (1, ))
    assert_size_stride(arg213_1, (24, ), (1, ))
    assert_size_stride(arg214_1, (24, ), (1, ))
    assert_size_stride(arg215_1, (96, 24), (24, 1))
    assert_size_stride(arg216_1, (96, ), (1, ))
    assert_size_stride(arg217_1, (24, 96), (96, 1))
    assert_size_stride(arg218_1, (24, ), (1, ))
    assert_size_stride(arg219_1, (24, ), (1, ))
    assert_size_stride(arg220_1, (24, ), (1, ))
    assert_size_stride(arg221_1, (384, 384), (384, 1))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (384, ), (1, ))
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (768, 384), (384, 1))
    assert_size_stride(arg226_1, (384, 384), (384, 1))
    assert_size_stride(arg227_1, (384, 384), (384, 1))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (1536, 384), (384, 1))
    assert_size_stride(arg232_1, (1536, ), (1, ))
    assert_size_stride(arg233_1, (384, 1536), (1536, 1))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (24, ), (1, ))
    assert_size_stride(arg236_1, (24, ), (1, ))
    assert_size_stride(arg237_1, (48, 24), (24, 1))
    assert_size_stride(arg238_1, (24, 24), (24, 1))
    assert_size_stride(arg239_1, (24, 24), (24, 1))
    assert_size_stride(arg240_1, (24, ), (1, ))
    assert_size_stride(arg241_1, (24, ), (1, ))
    assert_size_stride(arg242_1, (24, ), (1, ))
    assert_size_stride(arg243_1, (96, 24), (24, 1))
    assert_size_stride(arg244_1, (96, ), (1, ))
    assert_size_stride(arg245_1, (24, 96), (96, 1))
    assert_size_stride(arg246_1, (24, ), (1, ))
    assert_size_stride(arg247_1, (24, ), (1, ))
    assert_size_stride(arg248_1, (24, ), (1, ))
    assert_size_stride(arg249_1, (384, 384), (384, 1))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (768, 384), (384, 1))
    assert_size_stride(arg254_1, (384, 384), (384, 1))
    assert_size_stride(arg255_1, (384, 384), (384, 1))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (1536, 384), (384, 1))
    assert_size_stride(arg260_1, (1536, ), (1, ))
    assert_size_stride(arg261_1, (384, 1536), (1536, 1))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (24, ), (1, ))
    assert_size_stride(arg264_1, (24, ), (1, ))
    assert_size_stride(arg265_1, (48, 24), (24, 1))
    assert_size_stride(arg266_1, (24, 24), (24, 1))
    assert_size_stride(arg267_1, (24, 24), (24, 1))
    assert_size_stride(arg268_1, (24, ), (1, ))
    assert_size_stride(arg269_1, (24, ), (1, ))
    assert_size_stride(arg270_1, (24, ), (1, ))
    assert_size_stride(arg271_1, (96, 24), (24, 1))
    assert_size_stride(arg272_1, (96, ), (1, ))
    assert_size_stride(arg273_1, (24, 96), (96, 1))
    assert_size_stride(arg274_1, (24, ), (1, ))
    assert_size_stride(arg275_1, (24, ), (1, ))
    assert_size_stride(arg276_1, (24, ), (1, ))
    assert_size_stride(arg277_1, (384, 384), (384, 1))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (768, 384), (384, 1))
    assert_size_stride(arg282_1, (384, 384), (384, 1))
    assert_size_stride(arg283_1, (384, 384), (384, 1))
    assert_size_stride(arg284_1, (384, ), (1, ))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (384, ), (1, ))
    assert_size_stride(arg287_1, (1536, 384), (384, 1))
    assert_size_stride(arg288_1, (1536, ), (1, ))
    assert_size_stride(arg289_1, (384, 1536), (1536, 1))
    assert_size_stride(arg290_1, (384, ), (1, ))
    assert_size_stride(arg291_1, (24, ), (1, ))
    assert_size_stride(arg292_1, (24, ), (1, ))
    assert_size_stride(arg293_1, (48, 24), (24, 1))
    assert_size_stride(arg294_1, (24, 24), (24, 1))
    assert_size_stride(arg295_1, (24, 24), (24, 1))
    assert_size_stride(arg296_1, (24, ), (1, ))
    assert_size_stride(arg297_1, (24, ), (1, ))
    assert_size_stride(arg298_1, (24, ), (1, ))
    assert_size_stride(arg299_1, (96, 24), (24, 1))
    assert_size_stride(arg300_1, (96, ), (1, ))
    assert_size_stride(arg301_1, (24, 96), (96, 1))
    assert_size_stride(arg302_1, (24, ), (1, ))
    assert_size_stride(arg303_1, (24, ), (1, ))
    assert_size_stride(arg304_1, (24, ), (1, ))
    assert_size_stride(arg305_1, (384, 384), (384, 1))
    assert_size_stride(arg306_1, (384, ), (1, ))
    assert_size_stride(arg307_1, (384, ), (1, ))
    assert_size_stride(arg308_1, (384, ), (1, ))
    assert_size_stride(arg309_1, (768, 384), (384, 1))
    assert_size_stride(arg310_1, (384, 384), (384, 1))
    assert_size_stride(arg311_1, (384, 384), (384, 1))
    assert_size_stride(arg312_1, (384, ), (1, ))
    assert_size_stride(arg313_1, (384, ), (1, ))
    assert_size_stride(arg314_1, (384, ), (1, ))
    assert_size_stride(arg315_1, (1536, 384), (384, 1))
    assert_size_stride(arg316_1, (1536, ), (1, ))
    assert_size_stride(arg317_1, (384, 1536), (1536, 1))
    assert_size_stride(arg318_1, (384, ), (1, ))
    assert_size_stride(arg319_1, (24, ), (1, ))
    assert_size_stride(arg320_1, (24, ), (1, ))
    assert_size_stride(arg321_1, (48, 24), (24, 1))
    assert_size_stride(arg322_1, (24, 24), (24, 1))
    assert_size_stride(arg323_1, (24, 24), (24, 1))
    assert_size_stride(arg324_1, (24, ), (1, ))
    assert_size_stride(arg325_1, (24, ), (1, ))
    assert_size_stride(arg326_1, (24, ), (1, ))
    assert_size_stride(arg327_1, (96, 24), (24, 1))
    assert_size_stride(arg328_1, (96, ), (1, ))
    assert_size_stride(arg329_1, (24, 96), (96, 1))
    assert_size_stride(arg330_1, (24, ), (1, ))
    assert_size_stride(arg331_1, (24, ), (1, ))
    assert_size_stride(arg332_1, (24, ), (1, ))
    assert_size_stride(arg333_1, (384, 384), (384, 1))
    assert_size_stride(arg334_1, (384, ), (1, ))
    assert_size_stride(arg335_1, (384, ), (1, ))
    assert_size_stride(arg336_1, (384, ), (1, ))
    assert_size_stride(arg337_1, (768, 384), (384, 1))
    assert_size_stride(arg338_1, (384, 384), (384, 1))
    assert_size_stride(arg339_1, (384, 384), (384, 1))
    assert_size_stride(arg340_1, (384, ), (1, ))
    assert_size_stride(arg341_1, (384, ), (1, ))
    assert_size_stride(arg342_1, (384, ), (1, ))
    assert_size_stride(arg343_1, (1536, 384), (384, 1))
    assert_size_stride(arg344_1, (1536, ), (1, ))
    assert_size_stride(arg345_1, (384, 1536), (1536, 1))
    assert_size_stride(arg346_1, (384, ), (1, ))
    assert_size_stride(arg347_1, (384, ), (1, ))
    assert_size_stride(arg348_1, (384, ), (1, ))
    assert_size_stride(arg349_1, (1000, 384), (384, 1))
    assert_size_stride(arg350_1, (1000, ), (1, ))
    assert_size_stride(arg351_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((24, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg351_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg351_1
    del arg3_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg4_1, stride=(4, 4), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg4_1
    del buf1
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 196, 384), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg5_1
    del arg6_1
    buf7 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf6, (1568, 384), (384, 1), 0), reinterpret_tensor(arg7_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf7)
    del arg7_1
    del arg8_1
    buf8 = buf4; del buf4  # reuse
    buf9 = buf3; del buf3  # reuse
    buf11 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf14 = reinterpret_tensor(buf6, (1568, 16, 24), (384, 24, 1), 0); del buf6  # reuse
    cpp_fused_native_layer_norm_2(c_void_p(buf7.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()))
    del arg11_1
    del arg12_1
    buf15 = reinterpret_tensor(buf0, (25088, 48), (48, 1), 0); del buf0  # reuse
    # Source Nodes: [l__mod___blocks_0_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (25088, 24), (24, 1), 0), reinterpret_tensor(arg13_1, (24, 48), (1, 24), 0), out=buf15)
    del arg13_1
    buf16 = empty((1568, 4, 16, 6), device='cpu', dtype=torch.float32)
    buf17 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_3(c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = empty((6272, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf16, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf17, (6272, 6, 16), (96, 16, 1), 0), out=buf18)
    buf19 = empty_strided((1568, 4, 16, 1), (64, 16, 1, 100352), device='cpu', dtype=torch.float32)
    buf20 = reinterpret_tensor(buf18, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf18  # reuse
    buf21 = empty_strided((1568, 4, 16, 1), (64, 16, 1, 100352), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_mul_4(c_void_p(buf20.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = reinterpret_tensor(buf17, (25088, 24), (24, 1), 0); del buf17  # reuse
    # Source Nodes: [l__mod___blocks_0_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf14, (25088, 24), (24, 1), 0), reinterpret_tensor(arg14_1, (24, 24), (1, 24), 0), out=buf22)
    del arg14_1
    buf23 = buf20; del buf20  # reuse
    buf24 = reinterpret_tensor(buf14, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf14  # reuse
    cpp_fused__softmax_clone_5(c_void_p(buf23.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = reinterpret_tensor(buf22, (6272, 16, 6), (96, 6, 1), 0); del buf22  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf23, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf24, (6272, 16, 6), (96, 6, 1), 0), out=buf25)
    buf26 = reinterpret_tensor(buf24, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf24  # reuse
    cpp_fused_clone_6(c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    buf27 = reinterpret_tensor(buf25, (25088, 24), (24, 1), 0); del buf25  # reuse
    # Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf26, (25088, 24), (24, 1), 0), reinterpret_tensor(arg15_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf27)
    del arg15_1
    del arg16_1
    buf28 = buf12; del buf12  # reuse
    buf29 = buf11; del buf11  # reuse
    buf31 = reinterpret_tensor(buf26, (1568, 16, 24), (384, 24, 1), 0); del buf26  # reuse
    cpp_fused_add_native_layer_norm_7(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg17_1
    del arg18_1
    buf32 = empty((25088, 96), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf31, (25088, 24), (24, 1), 0), reinterpret_tensor(arg19_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf32)
    del arg19_1
    del arg20_1
    buf33 = reinterpret_tensor(buf32, (1568, 16, 96), (1536, 96, 1), 0); del buf32  # reuse
    cpp_fused_gelu_8(c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf31, (25088, 24), (24, 1), 0); del buf31  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf33, (25088, 96), (96, 1), 0), reinterpret_tensor(arg21_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf34)
    del arg21_1
    del arg22_1
    buf35 = buf29; del buf29  # reuse
    buf36 = buf28; del buf28  # reuse
    buf61 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf62 = empty_strided((1568, 16, 1), (16, 1, 25088), device='cpu', dtype=torch.float32)
    buf38 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf39 = reinterpret_tensor(buf16, (1568, 16, 24), (384, 24, 1), 0); del buf16  # reuse
    buf64 = empty((1568, 16, 24), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_9(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg10_1
    del arg1_1
    del arg23_1
    del arg24_1
    del arg2_1
    del arg39_1
    del arg40_1
    del arg9_1
    del buf8
    del buf9
    buf40 = buf7; del buf7  # reuse
    # Source Nodes: [l__mod___blocks_0_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf39, (1568, 384), (384, 1), 0), reinterpret_tensor(arg25_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf40)
    del arg25_1
    del arg26_1
    buf41 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf44 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_10(c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf44.data_ptr()))
    del arg27_1
    del arg28_1
    buf45 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (1576, 384), (384, 1), 0), reinterpret_tensor(arg29_1, (384, 768), (1, 384), 0), out=buf45)
    del arg29_1
    buf46 = empty((8, 6, 197, 64), device='cpu', dtype=torch.float32)
    buf47 = empty((8, 6, 64, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_11(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    buf48 = empty((48, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf46, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf47, (48, 64, 197), (12608, 197, 1), 0), out=buf48)
    buf49 = empty_strided((8, 6, 197, 1), (1182, 197, 1, 9456), device='cpu', dtype=torch.float32)
    buf50 = reinterpret_tensor(buf48, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf48  # reuse
    buf51 = empty_strided((8, 6, 197, 1), (1182, 197, 1, 9456), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_mul_12(c_void_p(buf50.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()))
    buf52 = reinterpret_tensor(buf47, (1576, 384), (384, 1), 0); del buf47  # reuse
    # Source Nodes: [l__mod___blocks_0_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (1576, 384), (384, 1), 0), reinterpret_tensor(arg30_1, (384, 384), (1, 384), 0), out=buf52)
    del arg30_1
    buf53 = buf50; del buf50  # reuse
    buf54 = reinterpret_tensor(buf44, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf44  # reuse
    cpp_fused__softmax_clone_13(c_void_p(buf53.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()))
    buf55 = reinterpret_tensor(buf52, (48, 197, 64), (12608, 64, 1), 0); del buf52  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf53, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf54, (48, 197, 64), (12608, 64, 1), 0), out=buf55)
    buf56 = reinterpret_tensor(buf54, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf54  # reuse
    cpp_fused_clone_14(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = reinterpret_tensor(buf55, (1576, 384), (384, 1), 0); del buf55  # reuse
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf56, (1576, 384), (384, 1), 0), reinterpret_tensor(arg31_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf57)
    del arg31_1
    del arg32_1
    buf58 = buf42; del buf42  # reuse
    buf59 = buf41; del buf41  # reuse
    cpp_fused_add_cat_native_layer_norm_15(c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()))
    buf65 = buf15; del buf15  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (25088, 24), (24, 1), 0), reinterpret_tensor(arg41_1, (24, 48), (1, 24), 0), out=buf65)
    del arg41_1
    buf66 = reinterpret_tensor(buf39, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf39  # reuse
    buf67 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_16(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()))
    buf68 = reinterpret_tensor(buf23, (6272, 16, 16), (256, 16, 1), 0); del buf23  # reuse
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf66, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf67, (6272, 6, 16), (96, 16, 1), 0), out=buf68)
    buf69 = buf21; del buf21  # reuse
    buf70 = reinterpret_tensor(buf68, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf68  # reuse
    buf71 = buf19; del buf19  # reuse
    cpp_fused__softmax_mul_17(c_void_p(buf70.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()))
    buf72 = reinterpret_tensor(buf67, (25088, 24), (24, 1), 0); del buf67  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (25088, 24), (24, 1), 0), reinterpret_tensor(arg42_1, (24, 24), (1, 24), 0), out=buf72)
    del arg42_1
    buf73 = buf70; del buf70  # reuse
    buf74 = reinterpret_tensor(buf64, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf64  # reuse
    cpp_fused__softmax_clone_18(c_void_p(buf73.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf72, (6272, 16, 6), (96, 6, 1), 0); del buf72  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf73, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf74, (6272, 16, 6), (96, 6, 1), 0), out=buf75)
    buf76 = reinterpret_tensor(buf74, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf74  # reuse
    cpp_fused_clone_19(c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = reinterpret_tensor(buf75, (25088, 24), (24, 1), 0); del buf75  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf76, (25088, 24), (24, 1), 0), reinterpret_tensor(arg43_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf77)
    del arg43_1
    del arg44_1
    buf78 = reinterpret_tensor(buf77, (1568, 16, 24), (384, 24, 1), 0); del buf77  # reuse
    buf79 = buf62; del buf62  # reuse
    buf80 = buf61; del buf61  # reuse
    buf82 = reinterpret_tensor(buf76, (1568, 16, 24), (384, 24, 1), 0); del buf76  # reuse
    cpp_fused_add_native_layer_norm_20(c_void_p(buf78.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    del arg0_1
    del arg45_1
    del arg46_1
    buf83 = reinterpret_tensor(buf33, (25088, 96), (96, 1), 0); del buf33  # reuse
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg48_1, reinterpret_tensor(buf82, (25088, 24), (24, 1), 0), reinterpret_tensor(arg47_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf83)
    del arg47_1
    del arg48_1
    buf84 = reinterpret_tensor(buf83, (1568, 16, 96), (1536, 96, 1), 0); del buf83  # reuse
    cpp_fused_gelu_21(c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf82, (25088, 24), (24, 1), 0); del buf82  # reuse
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf84, (25088, 96), (96, 1), 0), reinterpret_tensor(arg49_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf85)
    del arg49_1
    del arg50_1
    buf86 = buf80; del buf80  # reuse
    buf87 = buf79; del buf79  # reuse
    buf116 = buf36; del buf36  # reuse
    buf117 = buf35; del buf35  # reuse
    buf89 = reinterpret_tensor(buf56, (8, 197, 384), (75648, 384, 1), 0); del buf56  # reuse
    cpp_fused_add_cat_native_layer_norm_22(c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg33_1
    del arg34_1
    buf90 = empty((1576, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_17], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg36_1, reinterpret_tensor(buf89, (1576, 384), (384, 1), 0), reinterpret_tensor(arg35_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf90)
    del arg35_1
    del arg36_1
    buf91 = reinterpret_tensor(buf90, (8, 197, 1536), (302592, 1536, 1), 0); del buf90  # reuse
    cpp_fused_gelu_23(c_void_p(buf91.data_ptr()))
    buf92 = reinterpret_tensor(buf89, (1576, 384), (384, 1), 0); del buf89  # reuse
    # Source Nodes: [x_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf91, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg37_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf92)
    del arg37_1
    del arg38_1
    buf93 = reinterpret_tensor(buf92, (8, 197, 384), (75648, 384, 1), 0); del buf92  # reuse
    buf94 = reinterpret_tensor(buf34, (1568, 16, 24), (384, 24, 1), 0); del buf34  # reuse
    buf119 = reinterpret_tensor(buf27, (1568, 16, 24), (384, 24, 1), 0); del buf27  # reuse
    cpp_fused_add_cat_native_layer_norm_24(c_void_p(buf93.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf119.data_ptr()))
    del arg51_1
    del arg52_1
    del arg67_1
    del arg68_1
    buf95 = buf40; del buf40  # reuse
    # Source Nodes: [l__mod___blocks_1_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg54_1, reinterpret_tensor(buf94, (1568, 384), (384, 1), 0), reinterpret_tensor(arg53_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf95)
    del arg53_1
    del arg54_1
    buf96 = buf59; del buf59  # reuse
    buf97 = buf58; del buf58  # reuse
    buf99 = reinterpret_tensor(buf57, (8, 197, 384), (75648, 384, 1), 0); del buf57  # reuse
    cpp_fused_cat_native_layer_norm_25(c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()))
    del arg55_1
    del arg56_1
    buf100 = buf45; del buf45  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (1576, 384), (384, 1), 0), reinterpret_tensor(arg57_1, (384, 768), (1, 384), 0), out=buf100)
    del arg57_1
    buf101 = reinterpret_tensor(buf38, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf38  # reuse
    buf102 = reinterpret_tensor(buf46, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf46  # reuse
    cpp_fused_clone_26(c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = reinterpret_tensor(buf53, (48, 197, 197), (38809, 197, 1), 0); del buf53  # reuse
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf101, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf102, (48, 64, 197), (12608, 197, 1), 0), out=buf103)
    buf104 = buf51; del buf51  # reuse
    buf105 = reinterpret_tensor(buf103, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf103  # reuse
    buf106 = buf49; del buf49  # reuse
    cpp_fused__softmax_mul_27(c_void_p(buf105.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    buf107 = reinterpret_tensor(buf102, (1576, 384), (384, 1), 0); del buf102  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf99, (1576, 384), (384, 1), 0), reinterpret_tensor(arg58_1, (384, 384), (1, 384), 0), out=buf107)
    del arg58_1
    buf108 = buf105; del buf105  # reuse
    buf109 = reinterpret_tensor(buf99, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf99  # reuse
    cpp_fused__softmax_clone_28(c_void_p(buf108.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    buf110 = reinterpret_tensor(buf107, (48, 197, 64), (12608, 64, 1), 0); del buf107  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf109, (48, 197, 64), (12608, 64, 1), 0), out=buf110)
    buf111 = reinterpret_tensor(buf109, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf109  # reuse
    cpp_fused_clone_29(c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = reinterpret_tensor(buf110, (1576, 384), (384, 1), 0); del buf110  # reuse
    # Source Nodes: [x_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf111, (1576, 384), (384, 1), 0), reinterpret_tensor(arg59_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf112)
    del arg59_1
    del arg60_1
    buf113 = buf97; del buf97  # reuse
    buf114 = buf96; del buf96  # reuse
    cpp_fused_add_cat_native_layer_norm_30(c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    buf120 = buf65; del buf65  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (25088, 24), (24, 1), 0), reinterpret_tensor(arg69_1, (24, 48), (1, 24), 0), out=buf120)
    del arg69_1
    buf121 = reinterpret_tensor(buf94, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf94  # reuse
    buf122 = reinterpret_tensor(buf2, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf2  # reuse
    cpp_fused_clone_31(c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = reinterpret_tensor(buf73, (6272, 16, 16), (256, 16, 1), 0); del buf73  # reuse
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf121, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf122, (6272, 6, 16), (96, 16, 1), 0), out=buf123)
    buf124 = buf71; del buf71  # reuse
    buf125 = reinterpret_tensor(buf123, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf123  # reuse
    buf126 = buf69; del buf69  # reuse
    cpp_fused__softmax_mul_32(c_void_p(buf125.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()))
    buf127 = reinterpret_tensor(buf122, (25088, 24), (24, 1), 0); del buf122  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (25088, 24), (24, 1), 0), reinterpret_tensor(arg70_1, (24, 24), (1, 24), 0), out=buf127)
    del arg70_1
    buf128 = buf125; del buf125  # reuse
    buf129 = reinterpret_tensor(buf119, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf119  # reuse
    cpp_fused__softmax_clone_33(c_void_p(buf128.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = reinterpret_tensor(buf127, (6272, 16, 6), (96, 6, 1), 0); del buf127  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf128, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf129, (6272, 16, 6), (96, 6, 1), 0), out=buf130)
    buf131 = reinterpret_tensor(buf129, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf129  # reuse
    cpp_fused_clone_34(c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    buf132 = reinterpret_tensor(buf130, (25088, 24), (24, 1), 0); del buf130  # reuse
    # Source Nodes: [x_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf131, (25088, 24), (24, 1), 0), reinterpret_tensor(arg71_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf132)
    del arg71_1
    del arg72_1
    buf133 = buf87; del buf87  # reuse
    buf134 = buf86; del buf86  # reuse
    buf136 = reinterpret_tensor(buf131, (1568, 16, 24), (384, 24, 1), 0); del buf131  # reuse
    cpp_fused_add_native_layer_norm_35(c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg73_1
    del arg74_1
    buf137 = reinterpret_tensor(buf84, (25088, 96), (96, 1), 0); del buf84  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf136, (25088, 24), (24, 1), 0), reinterpret_tensor(arg75_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf137)
    del arg75_1
    del arg76_1
    buf138 = reinterpret_tensor(buf137, (1568, 16, 96), (1536, 96, 1), 0); del buf137  # reuse
    cpp_fused_gelu_36(c_void_p(buf138.data_ptr()))
    buf139 = reinterpret_tensor(buf136, (25088, 24), (24, 1), 0); del buf136  # reuse
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf138, (25088, 96), (96, 1), 0), reinterpret_tensor(arg77_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf139)
    del arg77_1
    del arg78_1
    buf140 = buf134; del buf134  # reuse
    buf141 = buf133; del buf133  # reuse
    buf170 = buf117; del buf117  # reuse
    buf171 = buf116; del buf116  # reuse
    buf143 = reinterpret_tensor(buf111, (8, 197, 384), (75648, 384, 1), 0); del buf111  # reuse
    cpp_fused_add_cat_native_layer_norm_37(c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf143.data_ptr()))
    del arg61_1
    del arg62_1
    buf144 = reinterpret_tensor(buf91, (1576, 1536), (1536, 1), 0); del buf91  # reuse
    # Source Nodes: [x_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf143, (1576, 384), (384, 1), 0), reinterpret_tensor(arg63_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf144)
    del arg63_1
    del arg64_1
    buf145 = reinterpret_tensor(buf144, (8, 197, 1536), (302592, 1536, 1), 0); del buf144  # reuse
    cpp_fused_gelu_38(c_void_p(buf145.data_ptr()))
    buf146 = reinterpret_tensor(buf143, (1576, 384), (384, 1), 0); del buf143  # reuse
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf145, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg65_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf146)
    del arg65_1
    del arg66_1
    buf147 = reinterpret_tensor(buf146, (8, 197, 384), (75648, 384, 1), 0); del buf146  # reuse
    buf148 = reinterpret_tensor(buf121, (1568, 16, 24), (384, 24, 1), 0); del buf121  # reuse
    buf173 = reinterpret_tensor(buf66, (1568, 16, 24), (384, 24, 1), 0); del buf66  # reuse
    cpp_fused_add_cat_native_layer_norm_39(c_void_p(buf147.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf173.data_ptr()))
    del arg79_1
    del arg80_1
    del arg95_1
    del arg96_1
    buf149 = buf95; del buf95  # reuse
    # Source Nodes: [l__mod___blocks_2_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf148, (1568, 384), (384, 1), 0), reinterpret_tensor(arg81_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf149)
    del arg81_1
    del arg82_1
    buf150 = buf114; del buf114  # reuse
    buf151 = buf113; del buf113  # reuse
    buf153 = buf93; del buf93  # reuse
    cpp_fused_cat_native_layer_norm_40(c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg83_1
    del arg84_1
    buf154 = buf100; del buf100  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (1576, 384), (384, 1), 0), reinterpret_tensor(arg85_1, (384, 768), (1, 384), 0), out=buf154)
    del arg85_1
    buf155 = reinterpret_tensor(buf112, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf112  # reuse
    buf156 = reinterpret_tensor(buf101, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf101  # reuse
    cpp_fused_clone_41(c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf108, (48, 197, 197), (38809, 197, 1), 0); del buf108  # reuse
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf155, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf156, (48, 64, 197), (12608, 197, 1), 0), out=buf157)
    buf158 = buf106; del buf106  # reuse
    buf159 = reinterpret_tensor(buf157, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf157  # reuse
    buf160 = buf104; del buf104  # reuse
    cpp_fused__softmax_mul_42(c_void_p(buf159.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()))
    buf161 = reinterpret_tensor(buf156, (1576, 384), (384, 1), 0); del buf156  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf153, (1576, 384), (384, 1), 0), reinterpret_tensor(arg86_1, (384, 384), (1, 384), 0), out=buf161)
    del arg86_1
    buf162 = buf159; del buf159  # reuse
    buf163 = reinterpret_tensor(buf153, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf153  # reuse
    cpp_fused__softmax_clone_43(c_void_p(buf162.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    buf164 = reinterpret_tensor(buf161, (48, 197, 64), (12608, 64, 1), 0); del buf161  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf162, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf163, (48, 197, 64), (12608, 64, 1), 0), out=buf164)
    buf165 = reinterpret_tensor(buf163, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf163  # reuse
    cpp_fused_clone_44(c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    buf166 = reinterpret_tensor(buf164, (1576, 384), (384, 1), 0); del buf164  # reuse
    # Source Nodes: [x_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf165, (1576, 384), (384, 1), 0), reinterpret_tensor(arg87_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf166)
    del arg87_1
    del arg88_1
    buf167 = buf151; del buf151  # reuse
    buf168 = buf150; del buf150  # reuse
    cpp_fused_add_cat_native_layer_norm_45(c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    buf174 = buf120; del buf120  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (25088, 24), (24, 1), 0), reinterpret_tensor(arg97_1, (24, 48), (1, 24), 0), out=buf174)
    del arg97_1
    buf175 = reinterpret_tensor(buf148, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf148  # reuse
    buf176 = empty((1568, 4, 6, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_46(c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    buf177 = reinterpret_tensor(buf128, (6272, 16, 16), (256, 16, 1), 0); del buf128  # reuse
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf175, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf176, (6272, 6, 16), (96, 16, 1), 0), out=buf177)
    buf178 = buf126; del buf126  # reuse
    buf179 = reinterpret_tensor(buf177, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf177  # reuse
    buf180 = buf124; del buf124  # reuse
    cpp_fused__softmax_mul_47(c_void_p(buf179.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    buf181 = reinterpret_tensor(buf176, (25088, 24), (24, 1), 0); del buf176  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (25088, 24), (24, 1), 0), reinterpret_tensor(arg98_1, (24, 24), (1, 24), 0), out=buf181)
    del arg98_1
    buf182 = buf179; del buf179  # reuse
    buf183 = reinterpret_tensor(buf173, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf173  # reuse
    cpp_fused__softmax_clone_48(c_void_p(buf182.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = reinterpret_tensor(buf181, (6272, 16, 6), (96, 6, 1), 0); del buf181  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf182, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf183, (6272, 16, 6), (96, 6, 1), 0), out=buf184)
    buf185 = reinterpret_tensor(buf183, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf183  # reuse
    cpp_fused_clone_49(c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = reinterpret_tensor(buf184, (25088, 24), (24, 1), 0); del buf184  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg100_1, reinterpret_tensor(buf185, (25088, 24), (24, 1), 0), reinterpret_tensor(arg99_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf186)
    del arg100_1
    del arg99_1
    buf187 = reinterpret_tensor(buf186, (1568, 16, 24), (384, 24, 1), 0); del buf186  # reuse
    buf188 = buf171; del buf171  # reuse
    buf189 = buf170; del buf170  # reuse
    buf191 = reinterpret_tensor(buf185, (1568, 16, 24), (384, 24, 1), 0); del buf185  # reuse
    cpp_fused_add_native_layer_norm_50(c_void_p(buf187.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()))
    del arg101_1
    del arg102_1
    buf192 = reinterpret_tensor(buf138, (25088, 96), (96, 1), 0); del buf138  # reuse
    # Source Nodes: [x_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf191, (25088, 24), (24, 1), 0), reinterpret_tensor(arg103_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf192)
    del arg103_1
    del arg104_1
    buf193 = reinterpret_tensor(buf192, (1568, 16, 96), (1536, 96, 1), 0); del buf192  # reuse
    cpp_fused_gelu_51(c_void_p(buf193.data_ptr()))
    buf194 = reinterpret_tensor(buf191, (25088, 24), (24, 1), 0); del buf191  # reuse
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg106_1, reinterpret_tensor(buf193, (25088, 96), (96, 1), 0), reinterpret_tensor(arg105_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf194)
    del arg105_1
    del arg106_1
    buf195 = buf189; del buf189  # reuse
    buf196 = buf188; del buf188  # reuse
    buf225 = buf141; del buf141  # reuse
    buf226 = buf140; del buf140  # reuse
    buf198 = reinterpret_tensor(buf165, (8, 197, 384), (75648, 384, 1), 0); del buf165  # reuse
    cpp_fused_add_cat_native_layer_norm_52(c_void_p(buf187.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf198.data_ptr()))
    del arg89_1
    del arg90_1
    buf199 = reinterpret_tensor(buf145, (1576, 1536), (1536, 1), 0); del buf145  # reuse
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf198, (1576, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf199)
    del arg91_1
    del arg92_1
    buf200 = reinterpret_tensor(buf199, (8, 197, 1536), (302592, 1536, 1), 0); del buf199  # reuse
    cpp_fused_gelu_53(c_void_p(buf200.data_ptr()))
    buf201 = reinterpret_tensor(buf198, (1576, 384), (384, 1), 0); del buf198  # reuse
    # Source Nodes: [x_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg94_1, reinterpret_tensor(buf200, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg93_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf201)
    del arg93_1
    del arg94_1
    buf202 = reinterpret_tensor(buf201, (8, 197, 384), (75648, 384, 1), 0); del buf201  # reuse
    buf203 = reinterpret_tensor(buf85, (1568, 16, 24), (384, 24, 1), 0); del buf85  # reuse
    buf228 = buf78; del buf78  # reuse
    cpp_fused_add_cat_native_layer_norm_54(c_void_p(buf202.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg107_1
    del arg108_1
    del arg123_1
    del arg124_1
    buf204 = buf149; del buf149  # reuse
    # Source Nodes: [l__mod___blocks_3_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf203, (1568, 384), (384, 1), 0), reinterpret_tensor(arg109_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf204)
    del arg109_1
    del arg110_1
    buf205 = buf168; del buf168  # reuse
    buf206 = buf167; del buf167  # reuse
    buf208 = reinterpret_tensor(buf166, (8, 197, 384), (75648, 384, 1), 0); del buf166  # reuse
    cpp_fused_cat_native_layer_norm_55(c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()))
    del arg111_1
    del arg112_1
    buf209 = buf154; del buf154  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (1576, 384), (384, 1), 0), reinterpret_tensor(arg113_1, (384, 768), (1, 384), 0), out=buf209)
    del arg113_1
    buf210 = reinterpret_tensor(buf147, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf147  # reuse
    buf211 = reinterpret_tensor(buf155, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf155  # reuse
    cpp_fused_clone_56(c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    buf212 = reinterpret_tensor(buf162, (48, 197, 197), (38809, 197, 1), 0); del buf162  # reuse
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf210, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf211, (48, 64, 197), (12608, 197, 1), 0), out=buf212)
    buf213 = buf160; del buf160  # reuse
    buf214 = reinterpret_tensor(buf212, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf212  # reuse
    buf215 = buf158; del buf158  # reuse
    cpp_fused__softmax_mul_57(c_void_p(buf214.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = reinterpret_tensor(buf211, (1576, 384), (384, 1), 0); del buf211  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (1576, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 384), (1, 384), 0), out=buf216)
    del arg114_1
    buf217 = buf214; del buf214  # reuse
    buf218 = reinterpret_tensor(buf208, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf208  # reuse
    cpp_fused__softmax_clone_58(c_void_p(buf217.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf216, (48, 197, 64), (12608, 64, 1), 0); del buf216  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf217, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf218, (48, 197, 64), (12608, 64, 1), 0), out=buf219)
    buf220 = reinterpret_tensor(buf218, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf218  # reuse
    cpp_fused_clone_59(c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    buf221 = reinterpret_tensor(buf219, (1576, 384), (384, 1), 0); del buf219  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg116_1, reinterpret_tensor(buf220, (1576, 384), (384, 1), 0), reinterpret_tensor(arg115_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf221)
    del arg115_1
    del arg116_1
    buf222 = buf206; del buf206  # reuse
    buf223 = buf205; del buf205  # reuse
    cpp_fused_add_cat_native_layer_norm_60(c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    buf229 = buf174; del buf174  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (25088, 24), (24, 1), 0), reinterpret_tensor(arg125_1, (24, 48), (1, 24), 0), out=buf229)
    del arg125_1
    buf230 = reinterpret_tensor(buf203, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf203  # reuse
    buf231 = reinterpret_tensor(buf139, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf139  # reuse
    cpp_fused_clone_61(c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    buf232 = reinterpret_tensor(buf182, (6272, 16, 16), (256, 16, 1), 0); del buf182  # reuse
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf230, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf231, (6272, 6, 16), (96, 16, 1), 0), out=buf232)
    buf233 = buf180; del buf180  # reuse
    buf234 = reinterpret_tensor(buf232, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf232  # reuse
    buf235 = buf178; del buf178  # reuse
    cpp_fused__softmax_mul_62(c_void_p(buf234.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf235.data_ptr()))
    buf236 = reinterpret_tensor(buf231, (25088, 24), (24, 1), 0); del buf231  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (25088, 24), (24, 1), 0), reinterpret_tensor(arg126_1, (24, 24), (1, 24), 0), out=buf236)
    del arg126_1
    buf237 = buf234; del buf234  # reuse
    buf238 = reinterpret_tensor(buf228, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf228  # reuse
    cpp_fused__softmax_clone_63(c_void_p(buf237.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = reinterpret_tensor(buf236, (6272, 16, 6), (96, 6, 1), 0); del buf236  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf237, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf238, (6272, 16, 6), (96, 6, 1), 0), out=buf239)
    buf240 = reinterpret_tensor(buf238, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf238  # reuse
    cpp_fused_clone_64(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf239, (25088, 24), (24, 1), 0); del buf239  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf240, (25088, 24), (24, 1), 0), reinterpret_tensor(arg127_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf241)
    del arg127_1
    del arg128_1
    buf242 = buf226; del buf226  # reuse
    buf243 = buf225; del buf225  # reuse
    buf245 = reinterpret_tensor(buf240, (1568, 16, 24), (384, 24, 1), 0); del buf240  # reuse
    cpp_fused_add_native_layer_norm_65(c_void_p(buf187.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()))
    del arg129_1
    del arg130_1
    buf246 = reinterpret_tensor(buf193, (25088, 96), (96, 1), 0); del buf193  # reuse
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf245, (25088, 24), (24, 1), 0), reinterpret_tensor(arg131_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf246)
    del arg131_1
    del arg132_1
    buf247 = reinterpret_tensor(buf246, (1568, 16, 96), (1536, 96, 1), 0); del buf246  # reuse
    cpp_fused_gelu_66(c_void_p(buf247.data_ptr()))
    buf248 = reinterpret_tensor(buf245, (25088, 24), (24, 1), 0); del buf245  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg134_1, reinterpret_tensor(buf247, (25088, 96), (96, 1), 0), reinterpret_tensor(arg133_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf248)
    del arg133_1
    del arg134_1
    buf249 = buf243; del buf243  # reuse
    buf250 = buf242; del buf242  # reuse
    buf279 = buf196; del buf196  # reuse
    buf280 = buf195; del buf195  # reuse
    buf252 = reinterpret_tensor(buf220, (8, 197, 384), (75648, 384, 1), 0); del buf220  # reuse
    cpp_fused_add_cat_native_layer_norm_67(c_void_p(buf187.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf252.data_ptr()))
    del arg117_1
    del arg118_1
    buf253 = reinterpret_tensor(buf200, (1576, 1536), (1536, 1), 0); del buf200  # reuse
    # Source Nodes: [x_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf252, (1576, 384), (384, 1), 0), reinterpret_tensor(arg119_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf253)
    del arg119_1
    del arg120_1
    buf254 = reinterpret_tensor(buf253, (8, 197, 1536), (302592, 1536, 1), 0); del buf253  # reuse
    cpp_fused_gelu_68(c_void_p(buf254.data_ptr()))
    buf255 = reinterpret_tensor(buf252, (1576, 384), (384, 1), 0); del buf252  # reuse
    # Source Nodes: [x_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg122_1, reinterpret_tensor(buf254, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg121_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf255)
    del arg121_1
    del arg122_1
    buf256 = reinterpret_tensor(buf255, (8, 197, 384), (75648, 384, 1), 0); del buf255  # reuse
    buf257 = reinterpret_tensor(buf230, (1568, 16, 24), (384, 24, 1), 0); del buf230  # reuse
    buf282 = reinterpret_tensor(buf132, (1568, 16, 24), (384, 24, 1), 0); del buf132  # reuse
    cpp_fused_add_cat_native_layer_norm_69(c_void_p(buf256.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf282.data_ptr()))
    del arg135_1
    del arg136_1
    del arg151_1
    del arg152_1
    buf258 = buf204; del buf204  # reuse
    # Source Nodes: [l__mod___blocks_4_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf257, (1568, 384), (384, 1), 0), reinterpret_tensor(arg137_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf258)
    del arg137_1
    del arg138_1
    buf259 = buf223; del buf223  # reuse
    buf260 = buf222; del buf222  # reuse
    buf262 = reinterpret_tensor(buf221, (8, 197, 384), (75648, 384, 1), 0); del buf221  # reuse
    cpp_fused_cat_native_layer_norm_70(c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()))
    del arg139_1
    del arg140_1
    buf263 = buf209; del buf209  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf262, (1576, 384), (384, 1), 0), reinterpret_tensor(arg141_1, (384, 768), (1, 384), 0), out=buf263)
    del arg141_1
    buf264 = reinterpret_tensor(buf202, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf202  # reuse
    buf265 = reinterpret_tensor(buf210, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf210  # reuse
    cpp_fused_clone_71(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = reinterpret_tensor(buf217, (48, 197, 197), (38809, 197, 1), 0); del buf217  # reuse
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf264, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf265, (48, 64, 197), (12608, 197, 1), 0), out=buf266)
    buf267 = buf215; del buf215  # reuse
    buf268 = reinterpret_tensor(buf266, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf266  # reuse
    buf269 = buf213; del buf213  # reuse
    cpp_fused__softmax_mul_72(c_void_p(buf268.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf265, (1576, 384), (384, 1), 0); del buf265  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf262, (1576, 384), (384, 1), 0), reinterpret_tensor(arg142_1, (384, 384), (1, 384), 0), out=buf270)
    del arg142_1
    buf271 = buf268; del buf268  # reuse
    buf272 = reinterpret_tensor(buf262, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf262  # reuse
    cpp_fused__softmax_clone_73(c_void_p(buf271.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = reinterpret_tensor(buf270, (48, 197, 64), (12608, 64, 1), 0); del buf270  # reuse
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf271, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf272, (48, 197, 64), (12608, 64, 1), 0), out=buf273)
    buf274 = reinterpret_tensor(buf272, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf272  # reuse
    cpp_fused_clone_74(c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    buf275 = reinterpret_tensor(buf273, (1576, 384), (384, 1), 0); del buf273  # reuse
    # Source Nodes: [x_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf274, (1576, 384), (384, 1), 0), reinterpret_tensor(arg143_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf275)
    del arg143_1
    del arg144_1
    buf276 = buf260; del buf260  # reuse
    buf277 = buf259; del buf259  # reuse
    cpp_fused_add_cat_native_layer_norm_75(c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()))
    buf283 = buf229; del buf229  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (25088, 24), (24, 1), 0), reinterpret_tensor(arg153_1, (24, 48), (1, 24), 0), out=buf283)
    del arg153_1
    buf284 = reinterpret_tensor(buf257, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf257  # reuse
    buf285 = reinterpret_tensor(buf175, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf175  # reuse
    cpp_fused_clone_76(c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = reinterpret_tensor(buf237, (6272, 16, 16), (256, 16, 1), 0); del buf237  # reuse
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf285, (6272, 6, 16), (96, 16, 1), 0), out=buf286)
    buf287 = buf235; del buf235  # reuse
    buf288 = reinterpret_tensor(buf286, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf286  # reuse
    buf289 = buf233; del buf233  # reuse
    cpp_fused__softmax_mul_77(c_void_p(buf288.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf285, (25088, 24), (24, 1), 0); del buf285  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf282, (25088, 24), (24, 1), 0), reinterpret_tensor(arg154_1, (24, 24), (1, 24), 0), out=buf290)
    del arg154_1
    buf291 = buf288; del buf288  # reuse
    buf292 = reinterpret_tensor(buf282, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf282  # reuse
    cpp_fused__softmax_clone_78(c_void_p(buf291.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()))
    buf293 = reinterpret_tensor(buf290, (6272, 16, 6), (96, 6, 1), 0); del buf290  # reuse
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf291, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf292, (6272, 16, 6), (96, 6, 1), 0), out=buf293)
    buf294 = reinterpret_tensor(buf292, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf292  # reuse
    cpp_fused_clone_79(c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    buf295 = reinterpret_tensor(buf293, (25088, 24), (24, 1), 0); del buf293  # reuse
    # Source Nodes: [x_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg156_1, reinterpret_tensor(buf294, (25088, 24), (24, 1), 0), reinterpret_tensor(arg155_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf295)
    del arg155_1
    del arg156_1
    buf296 = reinterpret_tensor(buf295, (1568, 16, 24), (384, 24, 1), 0); del buf295  # reuse
    buf297 = buf280; del buf280  # reuse
    buf298 = buf279; del buf279  # reuse
    buf300 = reinterpret_tensor(buf294, (1568, 16, 24), (384, 24, 1), 0); del buf294  # reuse
    cpp_fused_add_native_layer_norm_80(c_void_p(buf296.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()))
    del arg157_1
    del arg158_1
    buf301 = reinterpret_tensor(buf247, (25088, 96), (96, 1), 0); del buf247  # reuse
    # Source Nodes: [x_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf300, (25088, 24), (24, 1), 0), reinterpret_tensor(arg159_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf301)
    del arg159_1
    del arg160_1
    buf302 = reinterpret_tensor(buf301, (1568, 16, 96), (1536, 96, 1), 0); del buf301  # reuse
    cpp_fused_gelu_81(c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf300, (25088, 24), (24, 1), 0); del buf300  # reuse
    # Source Nodes: [x_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg162_1, reinterpret_tensor(buf302, (25088, 96), (96, 1), 0), reinterpret_tensor(arg161_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf303)
    del arg161_1
    del arg162_1
    buf304 = buf298; del buf298  # reuse
    buf305 = buf297; del buf297  # reuse
    buf334 = buf250; del buf250  # reuse
    buf335 = buf249; del buf249  # reuse
    buf307 = reinterpret_tensor(buf274, (8, 197, 384), (75648, 384, 1), 0); del buf274  # reuse
    cpp_fused_add_cat_native_layer_norm_82(c_void_p(buf296.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf307.data_ptr()))
    del arg145_1
    del arg146_1
    buf308 = reinterpret_tensor(buf254, (1576, 1536), (1536, 1), 0); del buf254  # reuse
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf307, (1576, 384), (384, 1), 0), reinterpret_tensor(arg147_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf308)
    del arg147_1
    del arg148_1
    buf309 = reinterpret_tensor(buf308, (8, 197, 1536), (302592, 1536, 1), 0); del buf308  # reuse
    cpp_fused_gelu_83(c_void_p(buf309.data_ptr()))
    buf310 = reinterpret_tensor(buf307, (1576, 384), (384, 1), 0); del buf307  # reuse
    # Source Nodes: [x_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf309, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg149_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf310)
    del arg149_1
    del arg150_1
    buf311 = reinterpret_tensor(buf310, (8, 197, 384), (75648, 384, 1), 0); del buf310  # reuse
    buf312 = reinterpret_tensor(buf248, (1568, 16, 24), (384, 24, 1), 0); del buf248  # reuse
    buf337 = reinterpret_tensor(buf241, (1568, 16, 24), (384, 24, 1), 0); del buf241  # reuse
    cpp_fused_add_cat_native_layer_norm_84(c_void_p(buf311.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf337.data_ptr()))
    del arg163_1
    del arg164_1
    del arg179_1
    del arg180_1
    buf313 = buf258; del buf258  # reuse
    # Source Nodes: [l__mod___blocks_5_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf312, (1568, 384), (384, 1), 0), reinterpret_tensor(arg165_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf313)
    del arg165_1
    del arg166_1
    buf314 = buf277; del buf277  # reuse
    buf315 = buf276; del buf276  # reuse
    buf317 = reinterpret_tensor(buf275, (8, 197, 384), (75648, 384, 1), 0); del buf275  # reuse
    cpp_fused_cat_native_layer_norm_85(c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()))
    del arg167_1
    del arg168_1
    buf318 = buf263; del buf263  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (1576, 384), (384, 1), 0), reinterpret_tensor(arg169_1, (384, 768), (1, 384), 0), out=buf318)
    del arg169_1
    buf319 = reinterpret_tensor(buf256, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf256  # reuse
    buf320 = reinterpret_tensor(buf264, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf264  # reuse
    cpp_fused_clone_86(c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    buf321 = reinterpret_tensor(buf271, (48, 197, 197), (38809, 197, 1), 0); del buf271  # reuse
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf319, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf320, (48, 64, 197), (12608, 197, 1), 0), out=buf321)
    buf322 = buf269; del buf269  # reuse
    buf323 = reinterpret_tensor(buf321, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf321  # reuse
    buf324 = buf267; del buf267  # reuse
    cpp_fused__softmax_mul_87(c_void_p(buf323.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()))
    buf325 = reinterpret_tensor(buf320, (1576, 384), (384, 1), 0); del buf320  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf317, (1576, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 384), (1, 384), 0), out=buf325)
    del arg170_1
    buf326 = buf323; del buf323  # reuse
    buf327 = reinterpret_tensor(buf317, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf317  # reuse
    cpp_fused__softmax_clone_88(c_void_p(buf326.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()))
    buf328 = reinterpret_tensor(buf325, (48, 197, 64), (12608, 64, 1), 0); del buf325  # reuse
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf326, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf327, (48, 197, 64), (12608, 64, 1), 0), out=buf328)
    buf329 = reinterpret_tensor(buf327, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf327  # reuse
    cpp_fused_clone_89(c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    buf330 = reinterpret_tensor(buf328, (1576, 384), (384, 1), 0); del buf328  # reuse
    # Source Nodes: [x_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf329, (1576, 384), (384, 1), 0), reinterpret_tensor(arg171_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf330)
    del arg171_1
    del arg172_1
    buf331 = buf315; del buf315  # reuse
    buf332 = buf314; del buf314  # reuse
    cpp_fused_add_cat_native_layer_norm_90(c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    buf338 = buf283; del buf283  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (25088, 24), (24, 1), 0), reinterpret_tensor(arg181_1, (24, 48), (1, 24), 0), out=buf338)
    del arg181_1
    buf339 = reinterpret_tensor(buf312, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf312  # reuse
    buf340 = reinterpret_tensor(buf194, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf194  # reuse
    cpp_fused_clone_91(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    buf341 = reinterpret_tensor(buf291, (6272, 16, 16), (256, 16, 1), 0); del buf291  # reuse
    # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf339, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf340, (6272, 6, 16), (96, 16, 1), 0), out=buf341)
    buf342 = buf289; del buf289  # reuse
    buf343 = reinterpret_tensor(buf341, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf341  # reuse
    buf344 = buf287; del buf287  # reuse
    cpp_fused__softmax_mul_92(c_void_p(buf343.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()))
    buf345 = reinterpret_tensor(buf340, (25088, 24), (24, 1), 0); del buf340  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf337, (25088, 24), (24, 1), 0), reinterpret_tensor(arg182_1, (24, 24), (1, 24), 0), out=buf345)
    del arg182_1
    buf346 = buf343; del buf343  # reuse
    buf347 = reinterpret_tensor(buf337, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf337  # reuse
    cpp_fused__softmax_clone_93(c_void_p(buf346.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    buf348 = reinterpret_tensor(buf345, (6272, 16, 6), (96, 6, 1), 0); del buf345  # reuse
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf346, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf347, (6272, 16, 6), (96, 6, 1), 0), out=buf348)
    buf349 = reinterpret_tensor(buf347, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf347  # reuse
    cpp_fused_clone_94(c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()))
    buf350 = reinterpret_tensor(buf348, (25088, 24), (24, 1), 0); del buf348  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg184_1, reinterpret_tensor(buf349, (25088, 24), (24, 1), 0), reinterpret_tensor(arg183_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf350)
    del arg183_1
    del arg184_1
    buf351 = buf335; del buf335  # reuse
    buf352 = buf334; del buf334  # reuse
    buf354 = reinterpret_tensor(buf349, (1568, 16, 24), (384, 24, 1), 0); del buf349  # reuse
    cpp_fused_add_native_layer_norm_95(c_void_p(buf296.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf354.data_ptr()))
    del arg185_1
    del arg186_1
    buf355 = reinterpret_tensor(buf302, (25088, 96), (96, 1), 0); del buf302  # reuse
    # Source Nodes: [x_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf354, (25088, 24), (24, 1), 0), reinterpret_tensor(arg187_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf355)
    del arg187_1
    del arg188_1
    buf356 = reinterpret_tensor(buf355, (1568, 16, 96), (1536, 96, 1), 0); del buf355  # reuse
    cpp_fused_gelu_96(c_void_p(buf356.data_ptr()))
    buf357 = reinterpret_tensor(buf354, (25088, 24), (24, 1), 0); del buf354  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf356, (25088, 96), (96, 1), 0), reinterpret_tensor(arg189_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf357)
    del arg189_1
    del arg190_1
    buf358 = buf352; del buf352  # reuse
    buf359 = buf351; del buf351  # reuse
    buf388 = buf305; del buf305  # reuse
    buf389 = buf304; del buf304  # reuse
    buf361 = reinterpret_tensor(buf329, (8, 197, 384), (75648, 384, 1), 0); del buf329  # reuse
    cpp_fused_add_cat_native_layer_norm_97(c_void_p(buf296.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf361.data_ptr()))
    del arg173_1
    del arg174_1
    buf362 = reinterpret_tensor(buf309, (1576, 1536), (1536, 1), 0); del buf309  # reuse
    # Source Nodes: [x_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf361, (1576, 384), (384, 1), 0), reinterpret_tensor(arg175_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf362)
    del arg175_1
    del arg176_1
    buf363 = reinterpret_tensor(buf362, (8, 197, 1536), (302592, 1536, 1), 0); del buf362  # reuse
    cpp_fused_gelu_98(c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf361, (1576, 384), (384, 1), 0); del buf361  # reuse
    # Source Nodes: [x_111], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf363, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg177_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf364)
    del arg177_1
    del arg178_1
    buf365 = reinterpret_tensor(buf364, (8, 197, 384), (75648, 384, 1), 0); del buf364  # reuse
    buf366 = reinterpret_tensor(buf339, (1568, 16, 24), (384, 24, 1), 0); del buf339  # reuse
    buf391 = buf187; del buf187  # reuse
    cpp_fused_add_cat_native_layer_norm_99(c_void_p(buf365.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf391.data_ptr()))
    del arg191_1
    del arg192_1
    del arg207_1
    del arg208_1
    buf367 = buf313; del buf313  # reuse
    # Source Nodes: [l__mod___blocks_6_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf366, (1568, 384), (384, 1), 0), reinterpret_tensor(arg193_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf367)
    del arg193_1
    del arg194_1
    buf368 = buf332; del buf332  # reuse
    buf369 = buf331; del buf331  # reuse
    buf371 = reinterpret_tensor(buf330, (8, 197, 384), (75648, 384, 1), 0); del buf330  # reuse
    cpp_fused_cat_native_layer_norm_100(c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf371.data_ptr()))
    del arg195_1
    del arg196_1
    buf372 = buf318; del buf318  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (1576, 384), (384, 1), 0), reinterpret_tensor(arg197_1, (384, 768), (1, 384), 0), out=buf372)
    del arg197_1
    buf373 = reinterpret_tensor(buf311, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf311  # reuse
    buf374 = reinterpret_tensor(buf319, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf319  # reuse
    cpp_fused_clone_101(c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()))
    buf375 = reinterpret_tensor(buf326, (48, 197, 197), (38809, 197, 1), 0); del buf326  # reuse
    # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf373, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf374, (48, 64, 197), (12608, 197, 1), 0), out=buf375)
    buf376 = buf324; del buf324  # reuse
    buf377 = reinterpret_tensor(buf375, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf375  # reuse
    buf378 = buf322; del buf322  # reuse
    cpp_fused__softmax_mul_102(c_void_p(buf377.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = reinterpret_tensor(buf374, (1576, 384), (384, 1), 0); del buf374  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf371, (1576, 384), (384, 1), 0), reinterpret_tensor(arg198_1, (384, 384), (1, 384), 0), out=buf379)
    del arg198_1
    buf380 = buf377; del buf377  # reuse
    buf381 = reinterpret_tensor(buf371, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf371  # reuse
    cpp_fused__softmax_clone_103(c_void_p(buf380.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = reinterpret_tensor(buf379, (48, 197, 64), (12608, 64, 1), 0); del buf379  # reuse
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf380, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf381, (48, 197, 64), (12608, 64, 1), 0), out=buf382)
    buf383 = reinterpret_tensor(buf381, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf381  # reuse
    cpp_fused_clone_104(c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    buf384 = reinterpret_tensor(buf382, (1576, 384), (384, 1), 0); del buf382  # reuse
    # Source Nodes: [x_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg200_1, reinterpret_tensor(buf383, (1576, 384), (384, 1), 0), reinterpret_tensor(arg199_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf384)
    del arg199_1
    del arg200_1
    buf385 = buf369; del buf369  # reuse
    buf386 = buf368; del buf368  # reuse
    cpp_fused_add_cat_native_layer_norm_105(c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    buf392 = buf338; del buf338  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (25088, 24), (24, 1), 0), reinterpret_tensor(arg209_1, (24, 48), (1, 24), 0), out=buf392)
    del arg209_1
    buf393 = reinterpret_tensor(buf366, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf366  # reuse
    buf394 = reinterpret_tensor(buf284, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf284  # reuse
    cpp_fused_clone_106(c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()))
    buf395 = reinterpret_tensor(buf346, (6272, 16, 16), (256, 16, 1), 0); del buf346  # reuse
    # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf393, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf394, (6272, 6, 16), (96, 16, 1), 0), out=buf395)
    buf396 = buf344; del buf344  # reuse
    buf397 = reinterpret_tensor(buf395, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf395  # reuse
    buf398 = buf342; del buf342  # reuse
    cpp_fused__softmax_mul_107(c_void_p(buf397.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf398.data_ptr()))
    buf399 = reinterpret_tensor(buf394, (25088, 24), (24, 1), 0); del buf394  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf391, (25088, 24), (24, 1), 0), reinterpret_tensor(arg210_1, (24, 24), (1, 24), 0), out=buf399)
    del arg210_1
    buf400 = buf397; del buf397  # reuse
    buf401 = reinterpret_tensor(buf391, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf391  # reuse
    cpp_fused__softmax_clone_108(c_void_p(buf400.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()))
    buf402 = reinterpret_tensor(buf399, (6272, 16, 6), (96, 6, 1), 0); del buf399  # reuse
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf400, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf401, (6272, 16, 6), (96, 6, 1), 0), out=buf402)
    buf403 = reinterpret_tensor(buf401, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf401  # reuse
    cpp_fused_clone_109(c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    buf404 = reinterpret_tensor(buf402, (25088, 24), (24, 1), 0); del buf402  # reuse
    # Source Nodes: [x_132], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg212_1, reinterpret_tensor(buf403, (25088, 24), (24, 1), 0), reinterpret_tensor(arg211_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf404)
    del arg211_1
    del arg212_1
    buf405 = reinterpret_tensor(buf404, (1568, 16, 24), (384, 24, 1), 0); del buf404  # reuse
    buf406 = buf389; del buf389  # reuse
    buf407 = buf388; del buf388  # reuse
    buf409 = reinterpret_tensor(buf403, (1568, 16, 24), (384, 24, 1), 0); del buf403  # reuse
    cpp_fused_add_native_layer_norm_110(c_void_p(buf405.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf409.data_ptr()))
    del arg213_1
    del arg214_1
    buf410 = reinterpret_tensor(buf356, (25088, 96), (96, 1), 0); del buf356  # reuse
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg216_1, reinterpret_tensor(buf409, (25088, 24), (24, 1), 0), reinterpret_tensor(arg215_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf410)
    del arg215_1
    del arg216_1
    buf411 = reinterpret_tensor(buf410, (1568, 16, 96), (1536, 96, 1), 0); del buf410  # reuse
    cpp_fused_gelu_111(c_void_p(buf411.data_ptr()))
    buf412 = reinterpret_tensor(buf409, (25088, 24), (24, 1), 0); del buf409  # reuse
    # Source Nodes: [x_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf411, (25088, 96), (96, 1), 0), reinterpret_tensor(arg217_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf412)
    del arg217_1
    del arg218_1
    buf413 = buf407; del buf407  # reuse
    buf414 = buf406; del buf406  # reuse
    buf443 = buf359; del buf359  # reuse
    buf444 = buf358; del buf358  # reuse
    buf416 = reinterpret_tensor(buf383, (8, 197, 384), (75648, 384, 1), 0); del buf383  # reuse
    cpp_fused_add_cat_native_layer_norm_112(c_void_p(buf405.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf416.data_ptr()))
    del arg201_1
    del arg202_1
    buf417 = reinterpret_tensor(buf363, (1576, 1536), (1536, 1), 0); del buf363  # reuse
    # Source Nodes: [x_125], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg204_1, reinterpret_tensor(buf416, (1576, 384), (384, 1), 0), reinterpret_tensor(arg203_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf417)
    del arg203_1
    del arg204_1
    buf418 = reinterpret_tensor(buf417, (8, 197, 1536), (302592, 1536, 1), 0); del buf417  # reuse
    cpp_fused_gelu_113(c_void_p(buf418.data_ptr()))
    buf419 = reinterpret_tensor(buf416, (1576, 384), (384, 1), 0); del buf416  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg206_1, reinterpret_tensor(buf418, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg205_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf419)
    del arg205_1
    del arg206_1
    buf420 = reinterpret_tensor(buf419, (8, 197, 384), (75648, 384, 1), 0); del buf419  # reuse
    buf421 = reinterpret_tensor(buf357, (1568, 16, 24), (384, 24, 1), 0); del buf357  # reuse
    buf446 = reinterpret_tensor(buf350, (1568, 16, 24), (384, 24, 1), 0); del buf350  # reuse
    cpp_fused_add_cat_native_layer_norm_114(c_void_p(buf420.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf446.data_ptr()))
    del arg219_1
    del arg220_1
    del arg235_1
    del arg236_1
    buf422 = buf367; del buf367  # reuse
    # Source Nodes: [l__mod___blocks_7_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg222_1, reinterpret_tensor(buf421, (1568, 384), (384, 1), 0), reinterpret_tensor(arg221_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf422)
    del arg221_1
    del arg222_1
    buf423 = buf386; del buf386  # reuse
    buf424 = buf385; del buf385  # reuse
    buf426 = reinterpret_tensor(buf384, (8, 197, 384), (75648, 384, 1), 0); del buf384  # reuse
    cpp_fused_cat_native_layer_norm_115(c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()))
    del arg223_1
    del arg224_1
    buf427 = buf372; del buf372  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf426, (1576, 384), (384, 1), 0), reinterpret_tensor(arg225_1, (384, 768), (1, 384), 0), out=buf427)
    del arg225_1
    buf428 = reinterpret_tensor(buf365, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf365  # reuse
    buf429 = reinterpret_tensor(buf373, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf373  # reuse
    cpp_fused_clone_116(c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    buf430 = reinterpret_tensor(buf380, (48, 197, 197), (38809, 197, 1), 0); del buf380  # reuse
    # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf428, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf429, (48, 64, 197), (12608, 197, 1), 0), out=buf430)
    buf431 = buf378; del buf378  # reuse
    buf432 = reinterpret_tensor(buf430, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf430  # reuse
    buf433 = buf376; del buf376  # reuse
    cpp_fused__softmax_mul_117(c_void_p(buf432.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()))
    buf434 = reinterpret_tensor(buf429, (1576, 384), (384, 1), 0); del buf429  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf426, (1576, 384), (384, 1), 0), reinterpret_tensor(arg226_1, (384, 384), (1, 384), 0), out=buf434)
    del arg226_1
    buf435 = buf432; del buf432  # reuse
    buf436 = reinterpret_tensor(buf426, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf426  # reuse
    cpp_fused__softmax_clone_118(c_void_p(buf435.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf436.data_ptr()))
    buf437 = reinterpret_tensor(buf434, (48, 197, 64), (12608, 64, 1), 0); del buf434  # reuse
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf435, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf436, (48, 197, 64), (12608, 64, 1), 0), out=buf437)
    buf438 = reinterpret_tensor(buf436, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf436  # reuse
    cpp_fused_clone_119(c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    buf439 = reinterpret_tensor(buf437, (1576, 384), (384, 1), 0); del buf437  # reuse
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg228_1, reinterpret_tensor(buf438, (1576, 384), (384, 1), 0), reinterpret_tensor(arg227_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf439)
    del arg227_1
    del arg228_1
    buf440 = buf424; del buf424  # reuse
    buf441 = buf423; del buf423  # reuse
    cpp_fused_add_cat_native_layer_norm_120(c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    buf447 = buf392; del buf392  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf446, (25088, 24), (24, 1), 0), reinterpret_tensor(arg237_1, (24, 48), (1, 24), 0), out=buf447)
    del arg237_1
    buf448 = reinterpret_tensor(buf421, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf421  # reuse
    buf449 = reinterpret_tensor(buf303, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf303  # reuse
    cpp_fused_clone_121(c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()))
    buf450 = reinterpret_tensor(buf400, (6272, 16, 16), (256, 16, 1), 0); del buf400  # reuse
    # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf448, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf449, (6272, 6, 16), (96, 16, 1), 0), out=buf450)
    buf451 = buf398; del buf398  # reuse
    buf452 = reinterpret_tensor(buf450, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf450  # reuse
    buf453 = buf396; del buf396  # reuse
    cpp_fused__softmax_mul_122(c_void_p(buf452.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf453.data_ptr()))
    buf454 = reinterpret_tensor(buf449, (25088, 24), (24, 1), 0); del buf449  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf446, (25088, 24), (24, 1), 0), reinterpret_tensor(arg238_1, (24, 24), (1, 24), 0), out=buf454)
    del arg238_1
    buf455 = buf452; del buf452  # reuse
    buf456 = reinterpret_tensor(buf446, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf446  # reuse
    cpp_fused__softmax_clone_123(c_void_p(buf455.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf456.data_ptr()))
    buf457 = reinterpret_tensor(buf454, (6272, 16, 6), (96, 6, 1), 0); del buf454  # reuse
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf455, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf456, (6272, 16, 6), (96, 6, 1), 0), out=buf457)
    buf458 = reinterpret_tensor(buf456, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf456  # reuse
    cpp_fused_clone_124(c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    buf459 = reinterpret_tensor(buf457, (25088, 24), (24, 1), 0); del buf457  # reuse
    # Source Nodes: [x_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg240_1, reinterpret_tensor(buf458, (25088, 24), (24, 1), 0), reinterpret_tensor(arg239_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf459)
    del arg239_1
    del arg240_1
    buf460 = buf444; del buf444  # reuse
    buf461 = buf443; del buf443  # reuse
    buf463 = reinterpret_tensor(buf458, (1568, 16, 24), (384, 24, 1), 0); del buf458  # reuse
    cpp_fused_add_native_layer_norm_125(c_void_p(buf405.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf463.data_ptr()))
    del arg241_1
    del arg242_1
    buf464 = reinterpret_tensor(buf411, (25088, 96), (96, 1), 0); del buf411  # reuse
    # Source Nodes: [x_152], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg244_1, reinterpret_tensor(buf463, (25088, 24), (24, 1), 0), reinterpret_tensor(arg243_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf464)
    del arg243_1
    del arg244_1
    buf465 = reinterpret_tensor(buf464, (1568, 16, 96), (1536, 96, 1), 0); del buf464  # reuse
    cpp_fused_gelu_126(c_void_p(buf465.data_ptr()))
    buf466 = reinterpret_tensor(buf463, (25088, 24), (24, 1), 0); del buf463  # reuse
    # Source Nodes: [x_156], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg246_1, reinterpret_tensor(buf465, (25088, 96), (96, 1), 0), reinterpret_tensor(arg245_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf466)
    del arg245_1
    del arg246_1
    buf467 = buf461; del buf461  # reuse
    buf468 = buf460; del buf460  # reuse
    buf497 = buf414; del buf414  # reuse
    buf498 = buf413; del buf413  # reuse
    buf470 = reinterpret_tensor(buf438, (8, 197, 384), (75648, 384, 1), 0); del buf438  # reuse
    cpp_fused_add_cat_native_layer_norm_127(c_void_p(buf405.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf470.data_ptr()))
    del arg229_1
    del arg230_1
    buf471 = reinterpret_tensor(buf418, (1576, 1536), (1536, 1), 0); del buf418  # reuse
    # Source Nodes: [x_143], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg232_1, reinterpret_tensor(buf470, (1576, 384), (384, 1), 0), reinterpret_tensor(arg231_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf471)
    del arg231_1
    del arg232_1
    buf472 = reinterpret_tensor(buf471, (8, 197, 1536), (302592, 1536, 1), 0); del buf471  # reuse
    cpp_fused_gelu_128(c_void_p(buf472.data_ptr()))
    buf473 = reinterpret_tensor(buf470, (1576, 384), (384, 1), 0); del buf470  # reuse
    # Source Nodes: [x_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg234_1, reinterpret_tensor(buf472, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg233_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf473)
    del arg233_1
    del arg234_1
    buf474 = reinterpret_tensor(buf473, (8, 197, 384), (75648, 384, 1), 0); del buf473  # reuse
    buf475 = reinterpret_tensor(buf448, (1568, 16, 24), (384, 24, 1), 0); del buf448  # reuse
    buf500 = buf296; del buf296  # reuse
    cpp_fused_add_cat_native_layer_norm_129(c_void_p(buf474.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf500.data_ptr()))
    del arg247_1
    del arg248_1
    del arg263_1
    del arg264_1
    buf476 = buf422; del buf422  # reuse
    # Source Nodes: [l__mod___blocks_8_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg250_1, reinterpret_tensor(buf475, (1568, 384), (384, 1), 0), reinterpret_tensor(arg249_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf476)
    del arg249_1
    del arg250_1
    buf477 = buf441; del buf441  # reuse
    buf478 = buf440; del buf440  # reuse
    buf480 = reinterpret_tensor(buf439, (8, 197, 384), (75648, 384, 1), 0); del buf439  # reuse
    cpp_fused_cat_native_layer_norm_130(c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf480.data_ptr()))
    del arg251_1
    del arg252_1
    buf481 = buf427; del buf427  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (1576, 384), (384, 1), 0), reinterpret_tensor(arg253_1, (384, 768), (1, 384), 0), out=buf481)
    del arg253_1
    buf482 = reinterpret_tensor(buf420, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf420  # reuse
    buf483 = reinterpret_tensor(buf428, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf428  # reuse
    cpp_fused_clone_131(c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()))
    buf484 = reinterpret_tensor(buf435, (48, 197, 197), (38809, 197, 1), 0); del buf435  # reuse
    # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf482, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf483, (48, 64, 197), (12608, 197, 1), 0), out=buf484)
    buf485 = buf433; del buf433  # reuse
    buf486 = reinterpret_tensor(buf484, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf484  # reuse
    buf487 = buf431; del buf431  # reuse
    cpp_fused__softmax_mul_132(c_void_p(buf486.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf487.data_ptr()))
    buf488 = reinterpret_tensor(buf483, (1576, 384), (384, 1), 0); del buf483  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf480, (1576, 384), (384, 1), 0), reinterpret_tensor(arg254_1, (384, 384), (1, 384), 0), out=buf488)
    del arg254_1
    buf489 = buf486; del buf486  # reuse
    buf490 = reinterpret_tensor(buf480, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf480  # reuse
    cpp_fused__softmax_clone_133(c_void_p(buf489.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf490.data_ptr()))
    buf491 = reinterpret_tensor(buf488, (48, 197, 64), (12608, 64, 1), 0); del buf488  # reuse
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf489, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf490, (48, 197, 64), (12608, 64, 1), 0), out=buf491)
    buf492 = reinterpret_tensor(buf490, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf490  # reuse
    cpp_fused_clone_134(c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()))
    buf493 = reinterpret_tensor(buf491, (1576, 384), (384, 1), 0); del buf491  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg256_1, reinterpret_tensor(buf492, (1576, 384), (384, 1), 0), reinterpret_tensor(arg255_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf493)
    del arg255_1
    del arg256_1
    buf494 = buf478; del buf478  # reuse
    buf495 = buf477; del buf477  # reuse
    cpp_fused_add_cat_native_layer_norm_135(c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    buf501 = buf447; del buf447  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf500, (25088, 24), (24, 1), 0), reinterpret_tensor(arg265_1, (24, 48), (1, 24), 0), out=buf501)
    del arg265_1
    buf502 = reinterpret_tensor(buf475, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf475  # reuse
    buf503 = reinterpret_tensor(buf393, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf393  # reuse
    cpp_fused_clone_136(c_void_p(buf501.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()))
    buf504 = reinterpret_tensor(buf455, (6272, 16, 16), (256, 16, 1), 0); del buf455  # reuse
    # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf502, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf503, (6272, 6, 16), (96, 16, 1), 0), out=buf504)
    buf505 = buf453; del buf453  # reuse
    buf506 = reinterpret_tensor(buf504, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf504  # reuse
    buf507 = buf451; del buf451  # reuse
    cpp_fused__softmax_mul_137(c_void_p(buf506.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf507.data_ptr()))
    buf508 = reinterpret_tensor(buf503, (25088, 24), (24, 1), 0); del buf503  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf500, (25088, 24), (24, 1), 0), reinterpret_tensor(arg266_1, (24, 24), (1, 24), 0), out=buf508)
    del arg266_1
    buf509 = buf506; del buf506  # reuse
    buf510 = reinterpret_tensor(buf500, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf500  # reuse
    cpp_fused__softmax_clone_138(c_void_p(buf509.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf510.data_ptr()))
    buf511 = reinterpret_tensor(buf508, (6272, 16, 6), (96, 6, 1), 0); del buf508  # reuse
    # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf509, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf510, (6272, 16, 6), (96, 6, 1), 0), out=buf511)
    buf512 = reinterpret_tensor(buf510, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf510  # reuse
    cpp_fused_clone_139(c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()))
    buf513 = reinterpret_tensor(buf511, (25088, 24), (24, 1), 0); del buf511  # reuse
    # Source Nodes: [x_168], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf512, (25088, 24), (24, 1), 0), reinterpret_tensor(arg267_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf513)
    del arg267_1
    del arg268_1
    buf514 = reinterpret_tensor(buf513, (1568, 16, 24), (384, 24, 1), 0); del buf513  # reuse
    buf515 = buf498; del buf498  # reuse
    buf516 = buf497; del buf497  # reuse
    buf518 = reinterpret_tensor(buf512, (1568, 16, 24), (384, 24, 1), 0); del buf512  # reuse
    cpp_fused_add_native_layer_norm_140(c_void_p(buf514.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf518.data_ptr()))
    del arg269_1
    del arg270_1
    buf519 = reinterpret_tensor(buf465, (25088, 96), (96, 1), 0); del buf465  # reuse
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg272_1, reinterpret_tensor(buf518, (25088, 24), (24, 1), 0), reinterpret_tensor(arg271_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf519)
    del arg271_1
    del arg272_1
    buf520 = reinterpret_tensor(buf519, (1568, 16, 96), (1536, 96, 1), 0); del buf519  # reuse
    cpp_fused_gelu_141(c_void_p(buf520.data_ptr()))
    buf521 = reinterpret_tensor(buf518, (25088, 24), (24, 1), 0); del buf518  # reuse
    # Source Nodes: [x_174], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg274_1, reinterpret_tensor(buf520, (25088, 96), (96, 1), 0), reinterpret_tensor(arg273_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf521)
    del arg273_1
    del arg274_1
    buf522 = buf516; del buf516  # reuse
    buf523 = buf515; del buf515  # reuse
    buf552 = buf468; del buf468  # reuse
    buf553 = buf467; del buf467  # reuse
    buf525 = reinterpret_tensor(buf492, (8, 197, 384), (75648, 384, 1), 0); del buf492  # reuse
    cpp_fused_add_cat_native_layer_norm_142(c_void_p(buf514.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf525.data_ptr()))
    del arg257_1
    del arg258_1
    buf526 = reinterpret_tensor(buf472, (1576, 1536), (1536, 1), 0); del buf472  # reuse
    # Source Nodes: [x_161], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg260_1, reinterpret_tensor(buf525, (1576, 384), (384, 1), 0), reinterpret_tensor(arg259_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf526)
    del arg259_1
    del arg260_1
    buf527 = reinterpret_tensor(buf526, (8, 197, 1536), (302592, 1536, 1), 0); del buf526  # reuse
    cpp_fused_gelu_143(c_void_p(buf527.data_ptr()))
    buf528 = reinterpret_tensor(buf525, (1576, 384), (384, 1), 0); del buf525  # reuse
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg262_1, reinterpret_tensor(buf527, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg261_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf528)
    del arg261_1
    del arg262_1
    buf529 = reinterpret_tensor(buf528, (8, 197, 384), (75648, 384, 1), 0); del buf528  # reuse
    buf530 = reinterpret_tensor(buf466, (1568, 16, 24), (384, 24, 1), 0); del buf466  # reuse
    buf555 = reinterpret_tensor(buf459, (1568, 16, 24), (384, 24, 1), 0); del buf459  # reuse
    cpp_fused_add_cat_native_layer_norm_144(c_void_p(buf529.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf555.data_ptr()))
    del arg275_1
    del arg276_1
    del arg291_1
    del arg292_1
    buf531 = buf476; del buf476  # reuse
    # Source Nodes: [l__mod___blocks_9_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf530, (1568, 384), (384, 1), 0), reinterpret_tensor(arg277_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf531)
    del arg277_1
    del arg278_1
    buf532 = buf495; del buf495  # reuse
    buf533 = buf494; del buf494  # reuse
    buf535 = reinterpret_tensor(buf493, (8, 197, 384), (75648, 384, 1), 0); del buf493  # reuse
    cpp_fused_cat_native_layer_norm_145(c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf535.data_ptr()))
    del arg279_1
    del arg280_1
    buf536 = buf481; del buf481  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf535, (1576, 384), (384, 1), 0), reinterpret_tensor(arg281_1, (384, 768), (1, 384), 0), out=buf536)
    del arg281_1
    buf537 = reinterpret_tensor(buf474, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf474  # reuse
    buf538 = reinterpret_tensor(buf482, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf482  # reuse
    cpp_fused_clone_146(c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()))
    buf539 = reinterpret_tensor(buf489, (48, 197, 197), (38809, 197, 1), 0); del buf489  # reuse
    # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf537, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf538, (48, 64, 197), (12608, 197, 1), 0), out=buf539)
    buf540 = buf487; del buf487  # reuse
    buf541 = reinterpret_tensor(buf539, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf539  # reuse
    buf542 = buf485; del buf485  # reuse
    cpp_fused__softmax_mul_147(c_void_p(buf541.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf542.data_ptr()))
    buf543 = reinterpret_tensor(buf538, (1576, 384), (384, 1), 0); del buf538  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf535, (1576, 384), (384, 1), 0), reinterpret_tensor(arg282_1, (384, 384), (1, 384), 0), out=buf543)
    del arg282_1
    buf544 = buf541; del buf541  # reuse
    buf545 = reinterpret_tensor(buf535, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf535  # reuse
    cpp_fused__softmax_clone_148(c_void_p(buf544.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf543.data_ptr()), c_void_p(buf545.data_ptr()))
    buf546 = reinterpret_tensor(buf543, (48, 197, 64), (12608, 64, 1), 0); del buf543  # reuse
    # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf544, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf545, (48, 197, 64), (12608, 64, 1), 0), out=buf546)
    buf547 = reinterpret_tensor(buf545, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf545  # reuse
    cpp_fused_clone_149(c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()))
    buf548 = reinterpret_tensor(buf546, (1576, 384), (384, 1), 0); del buf546  # reuse
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg284_1, reinterpret_tensor(buf547, (1576, 384), (384, 1), 0), reinterpret_tensor(arg283_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf548)
    del arg283_1
    del arg284_1
    buf549 = buf533; del buf533  # reuse
    buf550 = buf532; del buf532  # reuse
    cpp_fused_add_cat_native_layer_norm_150(c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()))
    buf556 = buf501; del buf501  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf555, (25088, 24), (24, 1), 0), reinterpret_tensor(arg293_1, (24, 48), (1, 24), 0), out=buf556)
    del arg293_1
    buf557 = reinterpret_tensor(buf530, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf530  # reuse
    buf558 = reinterpret_tensor(buf412, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf412  # reuse
    cpp_fused_clone_151(c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()))
    buf559 = reinterpret_tensor(buf509, (6272, 16, 16), (256, 16, 1), 0); del buf509  # reuse
    # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf557, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf558, (6272, 6, 16), (96, 16, 1), 0), out=buf559)
    buf560 = buf507; del buf507  # reuse
    buf561 = reinterpret_tensor(buf559, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf559  # reuse
    buf562 = buf505; del buf505  # reuse
    cpp_fused__softmax_mul_152(c_void_p(buf561.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf562.data_ptr()))
    buf563 = reinterpret_tensor(buf558, (25088, 24), (24, 1), 0); del buf558  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf555, (25088, 24), (24, 1), 0), reinterpret_tensor(arg294_1, (24, 24), (1, 24), 0), out=buf563)
    del arg294_1
    buf564 = buf561; del buf561  # reuse
    buf565 = reinterpret_tensor(buf555, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf555  # reuse
    cpp_fused__softmax_clone_153(c_void_p(buf564.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf565.data_ptr()))
    buf566 = reinterpret_tensor(buf563, (6272, 16, 6), (96, 6, 1), 0); del buf563  # reuse
    # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf564, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf565, (6272, 16, 6), (96, 6, 1), 0), out=buf566)
    buf567 = reinterpret_tensor(buf565, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf565  # reuse
    cpp_fused_clone_154(c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()))
    buf568 = reinterpret_tensor(buf566, (25088, 24), (24, 1), 0); del buf566  # reuse
    # Source Nodes: [x_186], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg296_1, reinterpret_tensor(buf567, (25088, 24), (24, 1), 0), reinterpret_tensor(arg295_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf568)
    del arg295_1
    del arg296_1
    buf569 = buf553; del buf553  # reuse
    buf570 = buf552; del buf552  # reuse
    buf572 = reinterpret_tensor(buf567, (1568, 16, 24), (384, 24, 1), 0); del buf567  # reuse
    cpp_fused_add_native_layer_norm_155(c_void_p(buf514.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf572.data_ptr()))
    del arg297_1
    del arg298_1
    buf573 = reinterpret_tensor(buf520, (25088, 96), (96, 1), 0); del buf520  # reuse
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg300_1, reinterpret_tensor(buf572, (25088, 24), (24, 1), 0), reinterpret_tensor(arg299_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf573)
    del arg299_1
    del arg300_1
    buf574 = reinterpret_tensor(buf573, (1568, 16, 96), (1536, 96, 1), 0); del buf573  # reuse
    cpp_fused_gelu_156(c_void_p(buf574.data_ptr()))
    buf575 = reinterpret_tensor(buf572, (25088, 24), (24, 1), 0); del buf572  # reuse
    # Source Nodes: [x_192], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg302_1, reinterpret_tensor(buf574, (25088, 96), (96, 1), 0), reinterpret_tensor(arg301_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf575)
    del arg301_1
    del arg302_1
    buf576 = buf570; del buf570  # reuse
    buf577 = buf569; del buf569  # reuse
    buf606 = buf523; del buf523  # reuse
    buf607 = buf522; del buf522  # reuse
    buf579 = reinterpret_tensor(buf547, (8, 197, 384), (75648, 384, 1), 0); del buf547  # reuse
    cpp_fused_add_cat_native_layer_norm_157(c_void_p(buf514.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf579.data_ptr()))
    del arg285_1
    del arg286_1
    buf580 = reinterpret_tensor(buf527, (1576, 1536), (1536, 1), 0); del buf527  # reuse
    # Source Nodes: [x_179], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg288_1, reinterpret_tensor(buf579, (1576, 384), (384, 1), 0), reinterpret_tensor(arg287_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf580)
    del arg287_1
    del arg288_1
    buf581 = reinterpret_tensor(buf580, (8, 197, 1536), (302592, 1536, 1), 0); del buf580  # reuse
    cpp_fused_gelu_158(c_void_p(buf581.data_ptr()))
    buf582 = reinterpret_tensor(buf579, (1576, 384), (384, 1), 0); del buf579  # reuse
    # Source Nodes: [x_183], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg290_1, reinterpret_tensor(buf581, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg289_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf582)
    del arg289_1
    del arg290_1
    buf583 = reinterpret_tensor(buf582, (8, 197, 384), (75648, 384, 1), 0); del buf582  # reuse
    buf584 = reinterpret_tensor(buf557, (1568, 16, 24), (384, 24, 1), 0); del buf557  # reuse
    buf609 = buf405; del buf405  # reuse
    cpp_fused_add_cat_native_layer_norm_159(c_void_p(buf583.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf609.data_ptr()))
    del arg303_1
    del arg304_1
    del arg319_1
    del arg320_1
    del buf576
    del buf577
    buf585 = buf531; del buf531  # reuse
    # Source Nodes: [l__mod___blocks_10_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg306_1, reinterpret_tensor(buf584, (1568, 384), (384, 1), 0), reinterpret_tensor(arg305_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf585)
    del arg305_1
    del arg306_1
    buf586 = buf550; del buf550  # reuse
    buf587 = buf549; del buf549  # reuse
    buf589 = reinterpret_tensor(buf548, (8, 197, 384), (75648, 384, 1), 0); del buf548  # reuse
    cpp_fused_cat_native_layer_norm_160(c_void_p(buf583.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf589.data_ptr()))
    del arg307_1
    del arg308_1
    buf590 = buf536; del buf536  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf589, (1576, 384), (384, 1), 0), reinterpret_tensor(arg309_1, (384, 768), (1, 384), 0), out=buf590)
    del arg309_1
    buf591 = reinterpret_tensor(buf529, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf529  # reuse
    buf592 = reinterpret_tensor(buf537, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf537  # reuse
    cpp_fused_clone_161(c_void_p(buf590.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()))
    buf593 = reinterpret_tensor(buf544, (48, 197, 197), (38809, 197, 1), 0); del buf544  # reuse
    # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf591, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf592, (48, 64, 197), (12608, 197, 1), 0), out=buf593)
    buf594 = buf542; del buf542  # reuse
    buf595 = reinterpret_tensor(buf593, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf593  # reuse
    buf596 = buf540; del buf540  # reuse
    cpp_fused__softmax_mul_162(c_void_p(buf595.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf596.data_ptr()))
    buf597 = reinterpret_tensor(buf592, (1576, 384), (384, 1), 0); del buf592  # reuse
    # Source Nodes: [l__mod___blocks_10_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf589, (1576, 384), (384, 1), 0), reinterpret_tensor(arg310_1, (384, 384), (1, 384), 0), out=buf597)
    del arg310_1
    buf598 = buf595; del buf595  # reuse
    buf599 = reinterpret_tensor(buf589, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf589  # reuse
    cpp_fused__softmax_clone_163(c_void_p(buf598.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf599.data_ptr()))
    buf600 = reinterpret_tensor(buf597, (48, 197, 64), (12608, 64, 1), 0); del buf597  # reuse
    # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf598, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf599, (48, 197, 64), (12608, 64, 1), 0), out=buf600)
    buf601 = reinterpret_tensor(buf599, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf599  # reuse
    cpp_fused_clone_164(c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()))
    buf602 = reinterpret_tensor(buf600, (1576, 384), (384, 1), 0); del buf600  # reuse
    # Source Nodes: [x_195], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg312_1, reinterpret_tensor(buf601, (1576, 384), (384, 1), 0), reinterpret_tensor(arg311_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf602)
    del arg311_1
    del arg312_1
    buf603 = buf587; del buf587  # reuse
    buf604 = buf586; del buf586  # reuse
    cpp_fused_add_cat_native_layer_norm_165(c_void_p(buf583.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()))
    buf610 = buf556; del buf556  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_in_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf609, (25088, 24), (24, 1), 0), reinterpret_tensor(arg321_1, (24, 48), (1, 24), 0), out=buf610)
    del arg321_1
    buf611 = reinterpret_tensor(buf584, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf584  # reuse
    buf612 = reinterpret_tensor(buf502, (1568, 4, 6, 16), (384, 96, 16, 1), 0); del buf502  # reuse
    cpp_fused_clone_166(c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()))
    del buf610
    buf613 = reinterpret_tensor(buf564, (6272, 16, 16), (256, 16, 1), 0); del buf564  # reuse
    # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf611, (6272, 16, 6), (96, 6, 1), 0), reinterpret_tensor(buf612, (6272, 6, 16), (96, 16, 1), 0), out=buf613)
    del buf611
    buf614 = buf562; del buf562  # reuse
    buf615 = reinterpret_tensor(buf613, (1568, 4, 16, 16), (1024, 256, 16, 1), 0); del buf613  # reuse
    buf616 = buf560; del buf560  # reuse
    cpp_fused__softmax_mul_167(c_void_p(buf615.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf616.data_ptr()))
    del buf614
    buf617 = reinterpret_tensor(buf612, (25088, 24), (24, 1), 0); del buf612  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_in_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf609, (25088, 24), (24, 1), 0), reinterpret_tensor(arg322_1, (24, 24), (1, 24), 0), out=buf617)
    del arg322_1
    buf618 = buf615; del buf615  # reuse
    buf619 = reinterpret_tensor(buf609, (1568, 4, 16, 6), (384, 96, 6, 1), 0); del buf609  # reuse
    cpp_fused__softmax_clone_168(c_void_p(buf618.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf619.data_ptr()))
    del buf616
    buf620 = reinterpret_tensor(buf617, (6272, 16, 6), (96, 6, 1), 0); del buf617  # reuse
    # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf618, (6272, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf619, (6272, 16, 6), (96, 6, 1), 0), out=buf620)
    del buf618
    buf621 = reinterpret_tensor(buf619, (1568, 16, 4, 6), (384, 24, 6, 1), 0); del buf619  # reuse
    cpp_fused_clone_169(c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()))
    buf622 = reinterpret_tensor(buf620, (25088, 24), (24, 1), 0); del buf620  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg324_1, reinterpret_tensor(buf621, (25088, 24), (24, 1), 0), reinterpret_tensor(arg323_1, (24, 24), (1, 24), 0), alpha=1, beta=1, out=buf622)
    del arg323_1
    del arg324_1
    buf623 = reinterpret_tensor(buf622, (1568, 16, 24), (384, 24, 1), 0); del buf622  # reuse
    buf624 = buf607; del buf607  # reuse
    buf625 = buf606; del buf606  # reuse
    buf627 = reinterpret_tensor(buf621, (1568, 16, 24), (384, 24, 1), 0); del buf621  # reuse
    cpp_fused_add_native_layer_norm_170(c_void_p(buf623.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf627.data_ptr()))
    del arg325_1
    del arg326_1
    del buf514
    del buf521
    del buf568
    buf628 = reinterpret_tensor(buf574, (25088, 96), (96, 1), 0); del buf574  # reuse
    # Source Nodes: [x_206], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg328_1, reinterpret_tensor(buf627, (25088, 24), (24, 1), 0), reinterpret_tensor(arg327_1, (24, 96), (1, 24), 0), alpha=1, beta=1, out=buf628)
    del arg327_1
    del arg328_1
    buf629 = reinterpret_tensor(buf628, (1568, 16, 96), (1536, 96, 1), 0); del buf628  # reuse
    cpp_fused_gelu_171(c_void_p(buf629.data_ptr()))
    buf630 = reinterpret_tensor(buf627, (25088, 24), (24, 1), 0); del buf627  # reuse
    # Source Nodes: [x_210], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg330_1, reinterpret_tensor(buf629, (25088, 96), (96, 1), 0), reinterpret_tensor(arg329_1, (96, 24), (1, 96), 0), alpha=1, beta=1, out=buf630)
    del arg329_1
    del arg330_1
    del buf629
    buf631 = buf625; del buf625  # reuse
    buf632 = buf624; del buf624  # reuse
    buf634 = reinterpret_tensor(buf601, (8, 197, 384), (75648, 384, 1), 0); del buf601  # reuse
    cpp_fused_add_cat_native_layer_norm_172(c_void_p(buf623.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf634.data_ptr()))
    del arg313_1
    del arg314_1
    buf635 = reinterpret_tensor(buf581, (1576, 1536), (1536, 1), 0); del buf581  # reuse
    # Source Nodes: [x_197], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg316_1, reinterpret_tensor(buf634, (1576, 384), (384, 1), 0), reinterpret_tensor(arg315_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf635)
    del arg315_1
    del arg316_1
    buf636 = reinterpret_tensor(buf635, (8, 197, 1536), (302592, 1536, 1), 0); del buf635  # reuse
    cpp_fused_gelu_173(c_void_p(buf636.data_ptr()))
    buf637 = reinterpret_tensor(buf634, (1576, 384), (384, 1), 0); del buf634  # reuse
    # Source Nodes: [x_201], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg318_1, reinterpret_tensor(buf636, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg317_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf637)
    del arg317_1
    del arg318_1
    buf638 = reinterpret_tensor(buf637, (8, 197, 384), (75648, 384, 1), 0); del buf637  # reuse
    buf639 = reinterpret_tensor(buf575, (1568, 16, 24), (384, 24, 1), 0); del buf575  # reuse
    cpp_fused_add_cat_native_layer_norm_174(c_void_p(buf638.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf630.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(buf639.data_ptr()))
    del arg331_1
    del arg332_1
    del buf585
    del buf623
    del buf631
    del buf632
    buf640 = reinterpret_tensor(buf630, (1568, 384), (384, 1), 0); del buf630  # reuse
    # Source Nodes: [l__mod___blocks_11_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg334_1, reinterpret_tensor(buf639, (1568, 384), (384, 1), 0), reinterpret_tensor(arg333_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf640)
    del arg333_1
    del arg334_1
    del buf639
    buf641 = buf604; del buf604  # reuse
    buf642 = buf603; del buf603  # reuse
    buf644 = reinterpret_tensor(buf602, (8, 197, 384), (75648, 384, 1), 0); del buf602  # reuse
    cpp_fused_cat_native_layer_norm_175(c_void_p(buf638.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf644.data_ptr()))
    del arg335_1
    del arg336_1
    buf645 = buf590; del buf590  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_out_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf644, (1576, 384), (384, 1), 0), reinterpret_tensor(arg337_1, (384, 768), (1, 384), 0), out=buf645)
    del arg337_1
    buf646 = reinterpret_tensor(buf583, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf583  # reuse
    buf647 = reinterpret_tensor(buf591, (8, 6, 64, 197), (75648, 12608, 197, 1), 0); del buf591  # reuse
    cpp_fused_clone_176(c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()))
    del buf645
    buf648 = reinterpret_tensor(buf598, (48, 197, 197), (38809, 197, 1), 0); del buf598  # reuse
    # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf646, (48, 197, 64), (12608, 64, 1), 0), reinterpret_tensor(buf647, (48, 64, 197), (12608, 197, 1), 0), out=buf648)
    del buf646
    buf649 = buf596; del buf596  # reuse
    buf650 = reinterpret_tensor(buf648, (8, 6, 197, 197), (232854, 38809, 197, 1), 0); del buf648  # reuse
    buf651 = buf594; del buf594  # reuse
    cpp_fused__softmax_mul_177(c_void_p(buf650.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf651.data_ptr()))
    del buf649
    buf652 = reinterpret_tensor(buf647, (1576, 384), (384, 1), 0); del buf647  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_out_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf644, (1576, 384), (384, 1), 0), reinterpret_tensor(arg338_1, (384, 384), (1, 384), 0), out=buf652)
    del arg338_1
    buf653 = buf650; del buf650  # reuse
    buf654 = reinterpret_tensor(buf644, (8, 6, 197, 64), (75648, 12608, 64, 1), 0); del buf644  # reuse
    cpp_fused__softmax_clone_178(c_void_p(buf653.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf654.data_ptr()))
    del buf651
    buf655 = reinterpret_tensor(buf652, (48, 197, 64), (12608, 64, 1), 0); del buf652  # reuse
    # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf653, (48, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf654, (48, 197, 64), (12608, 64, 1), 0), out=buf655)
    del buf653
    buf656 = reinterpret_tensor(buf654, (8, 197, 6, 64), (75648, 384, 64, 1), 0); del buf654  # reuse
    cpp_fused_clone_179(c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()))
    buf657 = reinterpret_tensor(buf655, (1576, 384), (384, 1), 0); del buf655  # reuse
    # Source Nodes: [x_213], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg340_1, reinterpret_tensor(buf656, (1576, 384), (384, 1), 0), reinterpret_tensor(arg339_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf657)
    del arg339_1
    del arg340_1
    buf658 = buf642; del buf642  # reuse
    buf659 = buf641; del buf641  # reuse
    buf661 = reinterpret_tensor(buf656, (8, 197, 384), (75648, 384, 1), 0); del buf656  # reuse
    cpp_fused_add_cat_native_layer_norm_180(c_void_p(buf638.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf661.data_ptr()))
    del arg341_1
    del arg342_1
    buf662 = reinterpret_tensor(buf636, (1576, 1536), (1536, 1), 0); del buf636  # reuse
    # Source Nodes: [x_215], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg344_1, reinterpret_tensor(buf661, (1576, 384), (384, 1), 0), reinterpret_tensor(arg343_1, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf662)
    del arg343_1
    del arg344_1
    buf663 = reinterpret_tensor(buf662, (8, 197, 1536), (302592, 1536, 1), 0); del buf662  # reuse
    cpp_fused_gelu_181(c_void_p(buf663.data_ptr()))
    buf664 = reinterpret_tensor(buf661, (1576, 384), (384, 1), 0); del buf661  # reuse
    # Source Nodes: [x_219], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg346_1, reinterpret_tensor(buf663, (1576, 1536), (1536, 1), 0), reinterpret_tensor(arg345_1, (1536, 384), (1, 1536), 0), alpha=1, beta=1, out=buf664)
    del arg345_1
    del arg346_1
    del buf663
    buf665 = reinterpret_tensor(buf664, (8, 197, 384), (75648, 384, 1), 0); del buf664  # reuse
    buf666 = buf659; del buf659  # reuse
    buf667 = buf658; del buf658  # reuse
    buf669 = empty((8, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_clone_native_layer_norm_182(c_void_p(buf665.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf669.data_ptr()))
    del arg347_1
    del arg348_1
    del buf638
    del buf640
    del buf657
    del buf665
    del buf666
    del buf667
    buf670 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_223, x_224], Original ATen: [aten.addmm, aten.clone]
    extern_kernels.addmm(arg350_1, buf669, reinterpret_tensor(arg349_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf670)
    del arg349_1
    del arg350_1
    return (buf670, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 24, 4, 4), (384, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 197, 384), (75648, 384, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((24, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((48, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((24, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((96, 24), (24, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((24, 96), (96, 1), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((1536, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((384, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('tnt_s_patch16_224', benchmark_compiled_module)
