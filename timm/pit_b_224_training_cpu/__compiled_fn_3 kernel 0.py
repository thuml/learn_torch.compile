
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (196L*x1) + (588L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (588L*x0))] = tmp0;
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


cpp_fused_cat_native_layer_norm_view_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(962L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(962);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (256L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(961L))) + (246016L*x0)), to_float_mask(tmp8));
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>((961L*x1) + (961L*x1_inner) + (static_cast<long>(((-1L) + x2)) % static_cast<long>(961L)))]; return masked_load(tmpbuf, to_float_mask(tmp8)); })();
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (246272L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_add_native_layer_norm_view_2 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1970176L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_8 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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


cpp_fused_add_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1970176L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (512L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<int>(257);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(256L))) + (131072L*x0)), to_float_mask(tmp10));
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp10));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp9)::blendv(tmp15, tmp9, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(512.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_add_native_layer_norm_view_12 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1052672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_view_18 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1052672L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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


cpp_fused_add_native_layer_norm_view_24 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1052672L); x0+=static_cast<long>(8L))
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
    }
}
''')


cpp_fused_cat_native_layer_norm_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (1024L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            return tmp8;
                        }
                        ;
                        auto tmp9 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp10 = tmp0 >= tmp3;
                        auto tmp11 = static_cast<int>(65);
                        auto tmp12 = tmp0 < tmp11;
                        auto tmp13 = [&]
                        {
                            auto tmp14 = masked_load(in_ptr2 + static_cast<long>(x2 + (1024L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(64L))) + (65536L*x0)), to_float_mask(tmp10));
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp13())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp13(), to_float_mask(tmp10));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp9)::blendv(tmp15, tmp9, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_add_native_layer_norm_view_31 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(532480L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_37 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(1024.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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


cpp_fused_native_layer_norm_select_42 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (66560L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (66560L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (66560L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (66560L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (66560L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp4 = out_ptr2[static_cast<long>(x0)];
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
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
''')


cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_43 = async_compile.cpp('''
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
                       float* out_ptr11,
                       float* out_ptr12)
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1024.0);
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (1024L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (1024L*x0)), static_cast<long>(16L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (1024L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (16L*x2) + (1024L*x0)), static_cast<long>(16L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (1024L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (16L*x2) + (1024L*x0)), static_cast<long>(16L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(520L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (1024L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (16L*x2) + (1024L*x0)), static_cast<long>(16L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2056L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (512L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(x1 + (8L*x2) + (512L*x0)), static_cast<long>(8L));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr10[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr10[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr11[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr11[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(256.0);
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
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7696L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr12[static_cast<long>(x2 + (64L*x1) + (256L*x0))];
                        out_ptr12[static_cast<long>(x1 + (4L*x2) + (256L*x0))] = tmp0;
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173 = args
    args.clear()
    assert_size_stride(primals_1, (1, 256, 31, 31), (246016, 961, 31, 1))
    assert_size_stride(primals_2, (1, 1, 256), (256, 256, 1))
    assert_size_stride(primals_3, (256, 3, 14, 14), (588, 196, 14, 1))
    assert_size_stride(primals_4, (256, ), (1, ))
    assert_size_stride(primals_5, (256, ), (1, ))
    assert_size_stride(primals_6, (256, ), (1, ))
    assert_size_stride(primals_7, (768, 256), (256, 1))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (256, 256), (256, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (1024, 256), (256, 1))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_15, (256, 1024), (1024, 1))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (768, 256), (256, 1))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (256, 256), (256, 1))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (1024, 256), (256, 1))
    assert_size_stride(primals_26, (1024, ), (1, ))
    assert_size_stride(primals_27, (256, 1024), (1024, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (768, 256), (256, 1))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (256, 256), (256, 1))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (1024, 256), (256, 1))
    assert_size_stride(primals_38, (1024, ), (1, ))
    assert_size_stride(primals_39, (256, 1024), (1024, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256), (256, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (512, ), (1, ))
    assert_size_stride(primals_47, (1536, 512), (512, 1))
    assert_size_stride(primals_48, (1536, ), (1, ))
    assert_size_stride(primals_49, (512, 512), (512, 1))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (2048, 512), (512, 1))
    assert_size_stride(primals_54, (2048, ), (1, ))
    assert_size_stride(primals_55, (512, 2048), (2048, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (512, ), (1, ))
    assert_size_stride(primals_59, (1536, 512), (512, 1))
    assert_size_stride(primals_60, (1536, ), (1, ))
    assert_size_stride(primals_61, (512, 512), (512, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (512, ), (1, ))
    assert_size_stride(primals_65, (2048, 512), (512, 1))
    assert_size_stride(primals_66, (2048, ), (1, ))
    assert_size_stride(primals_67, (512, 2048), (2048, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (512, ), (1, ))
    assert_size_stride(primals_71, (1536, 512), (512, 1))
    assert_size_stride(primals_72, (1536, ), (1, ))
    assert_size_stride(primals_73, (512, 512), (512, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, ), (1, ))
    assert_size_stride(primals_77, (2048, 512), (512, 1))
    assert_size_stride(primals_78, (2048, ), (1, ))
    assert_size_stride(primals_79, (512, 2048), (2048, 1))
    assert_size_stride(primals_80, (512, ), (1, ))
    assert_size_stride(primals_81, (512, ), (1, ))
    assert_size_stride(primals_82, (512, ), (1, ))
    assert_size_stride(primals_83, (1536, 512), (512, 1))
    assert_size_stride(primals_84, (1536, ), (1, ))
    assert_size_stride(primals_85, (512, 512), (512, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, ), (1, ))
    assert_size_stride(primals_89, (2048, 512), (512, 1))
    assert_size_stride(primals_90, (2048, ), (1, ))
    assert_size_stride(primals_91, (512, 2048), (2048, 1))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (512, ), (1, ))
    assert_size_stride(primals_94, (512, ), (1, ))
    assert_size_stride(primals_95, (1536, 512), (512, 1))
    assert_size_stride(primals_96, (1536, ), (1, ))
    assert_size_stride(primals_97, (512, 512), (512, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (512, ), (1, ))
    assert_size_stride(primals_101, (2048, 512), (512, 1))
    assert_size_stride(primals_102, (2048, ), (1, ))
    assert_size_stride(primals_103, (512, 2048), (2048, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, ), (1, ))
    assert_size_stride(primals_107, (1536, 512), (512, 1))
    assert_size_stride(primals_108, (1536, ), (1, ))
    assert_size_stride(primals_109, (512, 512), (512, 1))
    assert_size_stride(primals_110, (512, ), (1, ))
    assert_size_stride(primals_111, (512, ), (1, ))
    assert_size_stride(primals_112, (512, ), (1, ))
    assert_size_stride(primals_113, (2048, 512), (512, 1))
    assert_size_stride(primals_114, (2048, ), (1, ))
    assert_size_stride(primals_115, (512, 2048), (2048, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (1024, ), (1, ))
    assert_size_stride(primals_119, (1024, 512), (512, 1))
    assert_size_stride(primals_120, (1024, ), (1, ))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (1024, ), (1, ))
    assert_size_stride(primals_123, (3072, 1024), (1024, 1))
    assert_size_stride(primals_124, (3072, ), (1, ))
    assert_size_stride(primals_125, (1024, 1024), (1024, 1))
    assert_size_stride(primals_126, (1024, ), (1, ))
    assert_size_stride(primals_127, (1024, ), (1, ))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (4096, 1024), (1024, 1))
    assert_size_stride(primals_130, (4096, ), (1, ))
    assert_size_stride(primals_131, (1024, 4096), (4096, 1))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_135, (3072, 1024), (1024, 1))
    assert_size_stride(primals_136, (3072, ), (1, ))
    assert_size_stride(primals_137, (1024, 1024), (1024, 1))
    assert_size_stride(primals_138, (1024, ), (1, ))
    assert_size_stride(primals_139, (1024, ), (1, ))
    assert_size_stride(primals_140, (1024, ), (1, ))
    assert_size_stride(primals_141, (4096, 1024), (1024, 1))
    assert_size_stride(primals_142, (4096, ), (1, ))
    assert_size_stride(primals_143, (1024, 4096), (4096, 1))
    assert_size_stride(primals_144, (1024, ), (1, ))
    assert_size_stride(primals_145, (1024, ), (1, ))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (3072, 1024), (1024, 1))
    assert_size_stride(primals_148, (3072, ), (1, ))
    assert_size_stride(primals_149, (1024, 1024), (1024, 1))
    assert_size_stride(primals_150, (1024, ), (1, ))
    assert_size_stride(primals_151, (1024, ), (1, ))
    assert_size_stride(primals_152, (1024, ), (1, ))
    assert_size_stride(primals_153, (4096, 1024), (1024, 1))
    assert_size_stride(primals_154, (4096, ), (1, ))
    assert_size_stride(primals_155, (1024, 4096), (4096, 1))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (1024, ), (1, ))
    assert_size_stride(primals_158, (1024, ), (1, ))
    assert_size_stride(primals_159, (3072, 1024), (1024, 1))
    assert_size_stride(primals_160, (3072, ), (1, ))
    assert_size_stride(primals_161, (1024, 1024), (1024, 1))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (1024, ), (1, ))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_165, (4096, 1024), (1024, 1))
    assert_size_stride(primals_166, (4096, ), (1, ))
    assert_size_stride(primals_167, (1024, 4096), (4096, 1))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (1024, ), (1, ))
    assert_size_stride(primals_170, (1024, ), (1, ))
    assert_size_stride(primals_171, (1000, 1024), (1024, 1))
    assert_size_stride(primals_172, (1000, ), (1, ))
    assert_size_stride(primals_173, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((256, 3, 14, 14), (588, 1, 42, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_3.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_173
    del primals_3
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_4, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 256, 31, 31), (246016, 1, 7936, 256))
    del primals_4
    buf3 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    buf4 = empty((8, 962, 1), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf7 = reinterpret_tensor(buf5, (8, 962, 1), (962, 1, 1), 0); del buf5  # reuse
    buf8 = empty((7696, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_1(c_void_p(buf7.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf8.data_ptr()))
    del buf2
    del primals_1
    del primals_2
    del primals_6
    buf9 = empty((7696, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf8, reinterpret_tensor(primals_7, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf9)
    del primals_8
    # Source Nodes: [x_6], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf10 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 512))
    buf11 = buf10[0]
    buf12 = buf10[1]
    buf13 = buf10[2]
    buf14 = buf10[3]
    buf15 = buf10[6]
    buf16 = buf10[7]
    del buf10
    buf18 = empty((7696, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, reinterpret_tensor(buf11, (7696, 256), (256, 1), 0), reinterpret_tensor(primals_9, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf18)
    del primals_10
    buf19 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf22 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    buf23 = empty((7696, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_2(c_void_p(buf3.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del primals_12
    buf24 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf23, reinterpret_tensor(primals_13, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf24)
    del primals_14
    buf25 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_3(c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = empty((7696, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, buf25, reinterpret_tensor(primals_15, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf26)
    del primals_16
    buf27 = buf19; del buf19  # reuse
    buf28 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf30 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    buf31 = empty((7696, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_4(c_void_p(buf3.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_18
    buf32 = empty((7696, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, buf31, reinterpret_tensor(primals_19, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf32)
    del primals_20
    # Source Nodes: [x_18], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf33 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf32, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf32, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf32, (8, 4, 962, 64), (738816, 64, 768, 1), 512))
    buf34 = buf33[0]
    buf35 = buf33[1]
    buf36 = buf33[2]
    buf37 = buf33[3]
    buf38 = buf33[6]
    buf39 = buf33[7]
    del buf33
    buf41 = empty((7696, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, reinterpret_tensor(buf34, (7696, 256), (256, 1), 0), reinterpret_tensor(primals_21, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf41)
    del primals_22
    buf42 = buf27; del buf27  # reuse
    buf43 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf45 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    buf46 = empty((7696, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_5(c_void_p(buf3.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del primals_24
    buf47 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf46, reinterpret_tensor(primals_25, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf47)
    del primals_26
    buf48 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_6(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    buf49 = empty((7696, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_28, buf48, reinterpret_tensor(primals_27, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf49)
    del primals_28
    buf50 = reinterpret_tensor(buf49, (8, 962, 256), (246272, 256, 1), 0); del buf49  # reuse
    buf51 = buf42; del buf42  # reuse
    buf52 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf54 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    buf55 = empty((7696, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf50.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    del primals_30
    buf56 = empty((7696, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_32, buf55, reinterpret_tensor(primals_31, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf56)
    del primals_32
    # Source Nodes: [x_30], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf57 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf56, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf56, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf56, (8, 4, 962, 64), (738816, 64, 768, 1), 512))
    buf58 = buf57[0]
    buf59 = buf57[1]
    buf60 = buf57[2]
    buf61 = buf57[3]
    buf62 = buf57[6]
    buf63 = buf57[7]
    del buf57
    buf65 = buf41; del buf41  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_34, reinterpret_tensor(buf58, (7696, 256), (256, 1), 0), reinterpret_tensor(primals_33, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf65)
    del primals_34
    buf66 = buf51; del buf51  # reuse
    buf67 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf69 = reinterpret_tensor(buf26, (8, 962, 256), (246272, 256, 1), 0); del buf26  # reuse
    buf70 = buf18; del buf18  # reuse
    cpp_fused_add_native_layer_norm_view_8(c_void_p(buf50.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del buf66
    del primals_36
    buf71 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_38, buf70, reinterpret_tensor(primals_37, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf71)
    del primals_38
    buf72 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_9(c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = empty((7696, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_40, buf72, reinterpret_tensor(primals_39, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf73)
    del primals_40
    buf74 = reinterpret_tensor(buf73, (8, 962, 256), (246272, 256, 1), 0); del buf73  # reuse
    cpp_fused_add_10(c_void_p(buf74.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf65.data_ptr()))
    # Source Nodes: [x_47], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 256, 31, 31), (246272, 1, 7936, 256), 256), primals_41, primals_42, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256)
    assert_size_stride(buf75, (8, 512, 16, 16), (131072, 1, 8192, 512))
    del primals_42
    buf76 = empty((8, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [cls_tokens_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf74, (8, 256), (246272, 1), 0), reinterpret_tensor(primals_43, (256, 512), (1, 256), 0), out=buf76)
    buf77 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf78 = empty((8, 257, 1), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf81 = reinterpret_tensor(buf79, (8, 257, 1), (257, 1, 1), 0); del buf79  # reuse
    buf82 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_11(c_void_p(buf81.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf82.data_ptr()))
    del buf75
    del buf76
    del primals_44
    del primals_46
    buf83 = empty((2056, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, buf82, reinterpret_tensor(primals_47, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf83)
    del primals_48
    # Source Nodes: [x_51], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf84 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf83, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf83, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf83, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf85 = buf84[0]
    buf86 = buf84[1]
    buf87 = buf84[2]
    buf88 = buf84[3]
    buf89 = buf84[6]
    buf90 = buf84[7]
    del buf84
    buf92 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, reinterpret_tensor(buf85, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_49, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf92)
    del primals_50
    buf93 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf94 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf96 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf97 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_12(c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    del primals_52
    buf98 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf97, reinterpret_tensor(primals_53, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf98)
    del primals_54
    buf99 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_13(c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf99, reinterpret_tensor(primals_55, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf100)
    del primals_56
    buf101 = buf93; del buf93  # reuse
    buf102 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf104 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf105 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_14(c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del primals_58
    buf106 = empty((2056, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf105, reinterpret_tensor(primals_59, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf106)
    del primals_60
    # Source Nodes: [x_63], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf107 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf106, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf106, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf106, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf108 = buf107[0]
    buf109 = buf107[1]
    buf110 = buf107[2]
    buf111 = buf107[3]
    buf112 = buf107[6]
    buf113 = buf107[7]
    del buf107
    buf115 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_62, reinterpret_tensor(buf108, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_61, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf115)
    del primals_62
    buf116 = buf101; del buf101  # reuse
    buf117 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf119 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf120 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_15(c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()))
    del primals_64
    buf121 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_66, buf120, reinterpret_tensor(primals_65, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf121)
    del primals_66
    buf122 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_16(c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_68, buf122, reinterpret_tensor(primals_67, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf123)
    del primals_68
    buf124 = reinterpret_tensor(buf123, (8, 257, 512), (131584, 512, 1), 0); del buf123  # reuse
    buf125 = buf116; del buf116  # reuse
    buf126 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf128 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf129 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_17(c_void_p(buf124.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    del primals_70
    buf130 = empty((2056, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf129, reinterpret_tensor(primals_71, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf130)
    del primals_72
    # Source Nodes: [x_75], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf131 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf130, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf130, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf130, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf132 = buf131[0]
    buf133 = buf131[1]
    buf134 = buf131[2]
    buf135 = buf131[3]
    buf136 = buf131[6]
    buf137 = buf131[7]
    del buf131
    buf139 = buf92; del buf92  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, reinterpret_tensor(buf132, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_73, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf139)
    del primals_74
    buf140 = buf125; del buf125  # reuse
    buf141 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf143 = reinterpret_tensor(buf115, (8, 257, 512), (131584, 512, 1), 0); del buf115  # reuse
    buf144 = buf100; del buf100  # reuse
    cpp_fused_add_native_layer_norm_view_18(c_void_p(buf124.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()))
    del primals_76
    buf145 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_78, buf144, reinterpret_tensor(primals_77, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf145)
    del primals_78
    buf146 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_19(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    buf147 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf146, reinterpret_tensor(primals_79, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf147)
    del primals_80
    buf148 = buf140; del buf140  # reuse
    buf149 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf151 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf152 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_20(c_void_p(buf124.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    del primals_82
    buf153 = empty((2056, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf152, reinterpret_tensor(primals_83, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf153)
    del primals_84
    # Source Nodes: [x_87], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf154 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf153, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf153, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf153, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf155 = buf154[0]
    buf156 = buf154[1]
    buf157 = buf154[2]
    buf158 = buf154[3]
    buf159 = buf154[6]
    buf160 = buf154[7]
    del buf154
    buf162 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, reinterpret_tensor(buf155, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_85, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf162)
    del primals_86
    buf163 = buf148; del buf148  # reuse
    buf164 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf166 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf167 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_21(c_void_p(buf124.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del primals_88
    buf168 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf167, reinterpret_tensor(primals_89, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf168)
    del primals_90
    buf169 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_22(c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    buf170 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf169, reinterpret_tensor(primals_91, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf170)
    del primals_92
    buf171 = reinterpret_tensor(buf170, (8, 257, 512), (131584, 512, 1), 0); del buf170  # reuse
    buf172 = buf163; del buf163  # reuse
    buf173 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf175 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf176 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf171.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    del primals_94
    buf177 = empty((2056, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf176, reinterpret_tensor(primals_95, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf177)
    del primals_96
    # Source Nodes: [x_99], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf178 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf177, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf177, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf177, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf179 = buf178[0]
    buf180 = buf178[1]
    buf181 = buf178[2]
    buf182 = buf178[3]
    buf183 = buf178[6]
    buf184 = buf178[7]
    del buf178
    buf186 = buf162; del buf162  # reuse
    # Source Nodes: [x_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, reinterpret_tensor(buf179, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_97, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf186)
    del primals_98
    buf187 = buf172; del buf172  # reuse
    buf188 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf190 = reinterpret_tensor(buf147, (8, 257, 512), (131584, 512, 1), 0); del buf147  # reuse
    buf191 = buf139; del buf139  # reuse
    cpp_fused_add_native_layer_norm_view_24(c_void_p(buf171.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del primals_100
    buf192 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf191, reinterpret_tensor(primals_101, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf192)
    del primals_102
    buf193 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_25(c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    buf194 = reinterpret_tensor(buf124, (2056, 512), (512, 1), 0); del buf124  # reuse
    # Source Nodes: [x_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf193, reinterpret_tensor(primals_103, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf194)
    del primals_104
    buf195 = buf187; del buf187  # reuse
    buf196 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf198 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf199 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_26(c_void_p(buf171.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    del primals_106
    buf200 = empty((2056, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf199, reinterpret_tensor(primals_107, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf200)
    del primals_108
    # Source Nodes: [x_111], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf201 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf200, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf200, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf200, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf202 = buf201[0]
    buf203 = buf201[1]
    buf204 = buf201[2]
    buf205 = buf201[3]
    buf206 = buf201[6]
    buf207 = buf201[7]
    del buf201
    buf209 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, reinterpret_tensor(buf202, (2056, 512), (512, 1), 0), reinterpret_tensor(primals_109, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf209)
    del primals_110
    buf210 = buf195; del buf195  # reuse
    buf211 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf213 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    buf214 = empty((2056, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_27(c_void_p(buf171.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()))
    del buf210
    del primals_112
    buf215 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf214, reinterpret_tensor(primals_113, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf215)
    del primals_114
    buf216 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_28(c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    buf217 = empty((2056, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf216, reinterpret_tensor(primals_115, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf217)
    del primals_116
    buf218 = reinterpret_tensor(buf217, (8, 257, 512), (131584, 512, 1), 0); del buf217  # reuse
    cpp_fused_add_29(c_void_p(buf218.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf209.data_ptr()))
    # Source Nodes: [x_128], Original ATen: [aten.convolution]
    buf219 = extern_kernels.convolution(reinterpret_tensor(buf218, (8, 512, 16, 16), (131584, 1, 8192, 512), 512), primals_117, primals_118, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf219, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    del primals_118
    buf220 = empty((8, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [cls_tokens_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (8, 512), (131584, 1), 0), reinterpret_tensor(primals_119, (512, 1024), (1, 512), 0), out=buf220)
    buf221 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    buf222 = empty((8, 65, 1), device='cpu', dtype=torch.float32)
    buf223 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf225 = reinterpret_tensor(buf223, (8, 65, 1), (65, 1, 1), 0); del buf223  # reuse
    buf226 = empty((520, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_30(c_void_p(buf225.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf226.data_ptr()))
    del buf219
    del primals_120
    del primals_122
    buf227 = empty((520, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf226, reinterpret_tensor(primals_123, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf227)
    del primals_124
    # Source Nodes: [x_132], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf228 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf227, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf227, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf227, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    buf229 = buf228[0]
    buf230 = buf228[1]
    buf231 = buf228[2]
    buf232 = buf228[3]
    buf233 = buf228[6]
    buf234 = buf228[7]
    del buf228
    buf236 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_126, reinterpret_tensor(buf229, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_125, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf236)
    del primals_126
    buf237 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf238 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf240 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    buf241 = empty((520, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf221.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_128
    buf242 = empty((520, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf241, reinterpret_tensor(primals_129, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf242)
    del primals_130
    buf243 = empty((520, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_32(c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf243, reinterpret_tensor(primals_131, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf244)
    del primals_132
    buf245 = buf237; del buf237  # reuse
    buf246 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf248 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    buf249 = empty((520, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_33(c_void_p(buf221.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    del primals_134
    buf250 = empty((520, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_136, buf249, reinterpret_tensor(primals_135, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf250)
    del primals_136
    # Source Nodes: [x_144], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf251 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf250, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf250, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf250, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    buf252 = buf251[0]
    buf253 = buf251[1]
    buf254 = buf251[2]
    buf255 = buf251[3]
    buf256 = buf251[6]
    buf257 = buf251[7]
    del buf251
    buf259 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, reinterpret_tensor(buf252, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_137, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf259)
    del primals_138
    buf260 = buf245; del buf245  # reuse
    buf261 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf263 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    buf264 = empty((520, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_34(c_void_p(buf221.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    del primals_140
    buf265 = empty((520, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf264, reinterpret_tensor(primals_141, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf265)
    del primals_142
    buf266 = empty((520, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_35(c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf266, reinterpret_tensor(primals_143, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf267)
    del primals_144
    buf268 = reinterpret_tensor(buf267, (8, 65, 1024), (66560, 1024, 1), 0); del buf267  # reuse
    buf269 = buf260; del buf260  # reuse
    buf270 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf272 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    buf273 = empty((520, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_36(c_void_p(buf268.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    del primals_146
    buf274 = empty((520, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_148, buf273, reinterpret_tensor(primals_147, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf274)
    del primals_148
    # Source Nodes: [x_156], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf275 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf274, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf274, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf274, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    buf276 = buf275[0]
    buf277 = buf275[1]
    buf278 = buf275[2]
    buf279 = buf275[3]
    buf280 = buf275[6]
    buf281 = buf275[7]
    del buf275
    buf283 = buf259; del buf259  # reuse
    # Source Nodes: [x_158], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, reinterpret_tensor(buf276, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_149, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf283)
    del primals_150
    buf284 = buf269; del buf269  # reuse
    buf285 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf287 = reinterpret_tensor(buf244, (8, 65, 1024), (66560, 1024, 1), 0); del buf244  # reuse
    buf288 = buf236; del buf236  # reuse
    cpp_fused_add_native_layer_norm_view_37(c_void_p(buf268.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()))
    del primals_152
    buf289 = empty((520, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_161], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_154, buf288, reinterpret_tensor(primals_153, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf289)
    del primals_154
    buf290 = empty((520, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_38(c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()))
    buf291 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf290, reinterpret_tensor(primals_155, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf291)
    del primals_156
    buf292 = buf284; del buf284  # reuse
    buf293 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf295 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    buf296 = empty((520, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_39(c_void_p(buf268.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    del primals_158
    buf297 = empty((520, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_160, buf296, reinterpret_tensor(primals_159, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf297)
    del primals_160
    # Source Nodes: [x_168], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf298 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf297, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf297, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf297, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    buf299 = buf298[0]
    buf300 = buf298[1]
    buf301 = buf298[2]
    buf302 = buf298[3]
    buf303 = buf298[6]
    buf304 = buf298[7]
    del buf298
    buf306 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_162, reinterpret_tensor(buf299, (520, 1024), (1024, 1), 0), reinterpret_tensor(primals_161, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf306)
    del primals_162
    buf307 = buf292; del buf292  # reuse
    buf308 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf310 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    buf311 = empty((520, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_40(c_void_p(buf268.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    del buf307
    del primals_164
    buf312 = empty((520, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf311, reinterpret_tensor(primals_165, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf312)
    del primals_166
    buf313 = empty((520, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_41(c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = empty((520, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_168, buf313, reinterpret_tensor(primals_167, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf314)
    del primals_168
    buf315 = reinterpret_tensor(buf220, (8, 1, 1024), (1024, 8192, 1), 0); del buf220  # reuse
    buf316 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf317 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf319 = empty((8, 1, 1024), device='cpu', dtype=torch.float32)
    buf320 = empty((8, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_select_42(c_void_p(buf268.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del buf268
    del buf315
    del buf316
    del primals_170
    buf321 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_172, buf320, reinterpret_tensor(primals_171, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf321)
    del primals_172
    buf322 = reinterpret_tensor(buf317, (8, 1, 1), (1, 1, 1), 0); del buf317  # reuse
    buf323 = reinterpret_tensor(buf308, (8, 65, 1), (65, 1, 1), 0); del buf308  # reuse
    buf324 = reinterpret_tensor(buf314, (8, 16, 65, 64), (66560, 1, 1024, 16), 0); del buf314  # reuse
    buf325 = reinterpret_tensor(buf293, (8, 65, 1), (65, 1, 1), 0); del buf293  # reuse
    buf326 = reinterpret_tensor(buf285, (8, 65, 1), (65, 1, 1), 0); del buf285  # reuse
    buf327 = reinterpret_tensor(buf306, (8, 16, 65, 64), (66560, 1, 1024, 16), 0); del buf306  # reuse
    buf328 = reinterpret_tensor(buf270, (8, 65, 1), (65, 1, 1), 0); del buf270  # reuse
    buf329 = reinterpret_tensor(buf261, (8, 65, 1), (65, 1, 1), 0); del buf261  # reuse
    buf330 = reinterpret_tensor(buf291, (8, 16, 65, 64), (66560, 1, 1024, 16), 0); del buf291  # reuse
    buf331 = reinterpret_tensor(buf246, (8, 65, 1), (65, 1, 1), 0); del buf246  # reuse
    buf332 = reinterpret_tensor(buf238, (8, 65, 1), (65, 1, 1), 0); del buf238  # reuse
    buf333 = reinterpret_tensor(buf283, (8, 16, 65, 64), (66560, 1, 1024, 16), 0); del buf283  # reuse
    buf334 = reinterpret_tensor(buf211, (8, 257, 1), (257, 1, 1), 0); del buf211  # reuse
    buf335 = reinterpret_tensor(buf209, (8, 8, 257, 64), (131584, 1, 512, 8), 0); del buf209  # reuse
    buf336 = reinterpret_tensor(buf196, (8, 257, 1), (257, 1, 1), 0); del buf196  # reuse
    buf337 = reinterpret_tensor(buf188, (8, 257, 1), (257, 1, 1), 0); del buf188  # reuse
    buf338 = reinterpret_tensor(buf194, (8, 8, 257, 64), (131584, 1, 512, 8), 0); del buf194  # reuse
    buf339 = reinterpret_tensor(buf173, (8, 257, 1), (257, 1, 1), 0); del buf173  # reuse
    buf340 = reinterpret_tensor(buf164, (8, 257, 1), (257, 1, 1), 0); del buf164  # reuse
    buf341 = reinterpret_tensor(buf186, (8, 8, 257, 64), (131584, 1, 512, 8), 0); del buf186  # reuse
    buf342 = reinterpret_tensor(buf149, (8, 257, 1), (257, 1, 1), 0); del buf149  # reuse
    buf343 = reinterpret_tensor(buf141, (8, 257, 1), (257, 1, 1), 0); del buf141  # reuse
    buf344 = reinterpret_tensor(buf171, (8, 8, 257, 64), (131584, 1, 512, 8), 0); del buf171  # reuse
    buf345 = reinterpret_tensor(buf126, (8, 257, 1), (257, 1, 1), 0); del buf126  # reuse
    buf346 = reinterpret_tensor(buf117, (8, 257, 1), (257, 1, 1), 0); del buf117  # reuse
    buf347 = empty_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    buf348 = reinterpret_tensor(buf102, (8, 257, 1), (257, 1, 1), 0); del buf102  # reuse
    buf349 = reinterpret_tensor(buf94, (8, 257, 1), (257, 1, 1), 0); del buf94  # reuse
    buf350 = empty_strided((8, 8, 257, 64), (131584, 1, 512, 8), device='cpu', dtype=torch.float32)
    buf351 = reinterpret_tensor(buf67, (8, 962, 1), (962, 1, 1), 0); del buf67  # reuse
    buf352 = reinterpret_tensor(buf65, (8, 4, 962, 64), (246272, 1, 256, 4), 0); del buf65  # reuse
    buf353 = reinterpret_tensor(buf52, (8, 962, 1), (962, 1, 1), 0); del buf52  # reuse
    buf354 = reinterpret_tensor(buf43, (8, 962, 1), (962, 1, 1), 0); del buf43  # reuse
    buf355 = reinterpret_tensor(buf50, (8, 4, 962, 64), (246272, 1, 256, 4), 0); del buf50  # reuse
    buf356 = reinterpret_tensor(buf28, (8, 962, 1), (962, 1, 1), 0); del buf28  # reuse
    buf357 = reinterpret_tensor(buf20, (8, 962, 1), (962, 1, 1), 0); del buf20  # reuse
    buf358 = empty_strided((8, 4, 962, 64), (246272, 1, 256, 4), device='cpu', dtype=torch.float32)
    cpp_fused_add_detach_native_layer_norm_native_layer_norm_backward_43(c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf358.data_ptr()))
    return (buf321, buf0, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_157, primals_163, primals_169, buf1, buf3, buf4, buf7, buf8, reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf9, (8, 4, 962, 64), (738816, 64, 768, 1), 512), buf12, buf13, buf14, buf15, buf16, reinterpret_tensor(buf11, (7696, 256), (256, 1), 0), buf22, buf23, buf24, buf25, buf30, buf31, reinterpret_tensor(buf32, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf32, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf32, (8, 4, 962, 64), (738816, 64, 768, 1), 512), buf35, buf36, buf37, buf38, buf39, reinterpret_tensor(buf34, (7696, 256), (256, 1), 0), buf45, buf46, buf47, buf48, buf54, buf55, reinterpret_tensor(buf56, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf56, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf56, (8, 4, 962, 64), (738816, 64, 768, 1), 512), buf59, buf60, buf61, buf62, buf63, reinterpret_tensor(buf58, (7696, 256), (256, 1), 0), buf69, buf70, buf71, buf72, reinterpret_tensor(buf74, (8, 256, 31, 31), (246272, 1, 7936, 256), 256), reinterpret_tensor(buf74, (8, 256), (246272, 1), 0), buf77, buf78, buf81, buf82, reinterpret_tensor(buf83, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf83, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf83, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf86, buf87, buf88, buf89, buf90, reinterpret_tensor(buf85, (2056, 512), (512, 1), 0), buf96, buf97, buf98, buf99, buf104, buf105, reinterpret_tensor(buf106, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf106, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf106, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf109, buf110, buf111, buf112, buf113, reinterpret_tensor(buf108, (2056, 512), (512, 1), 0), buf119, buf120, buf121, buf122, buf128, buf129, reinterpret_tensor(buf130, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf130, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf130, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf133, buf134, buf135, buf136, buf137, reinterpret_tensor(buf132, (2056, 512), (512, 1), 0), buf143, buf144, buf145, buf146, buf151, buf152, reinterpret_tensor(buf153, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf153, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf153, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf156, buf157, buf158, buf159, buf160, reinterpret_tensor(buf155, (2056, 512), (512, 1), 0), buf166, buf167, buf168, buf169, buf175, buf176, reinterpret_tensor(buf177, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf177, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf177, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf180, buf181, buf182, buf183, buf184, reinterpret_tensor(buf179, (2056, 512), (512, 1), 0), buf190, buf191, buf192, buf193, buf198, buf199, reinterpret_tensor(buf200, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf200, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf200, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024), buf203, buf204, buf205, buf206, buf207, reinterpret_tensor(buf202, (2056, 512), (512, 1), 0), buf213, buf214, buf215, buf216, reinterpret_tensor(buf218, (8, 512, 16, 16), (131584, 1, 8192, 512), 512), reinterpret_tensor(buf218, (8, 512), (131584, 1), 0), buf221, buf222, buf225, buf226, reinterpret_tensor(buf227, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf227, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf227, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf230, buf231, buf232, buf233, buf234, reinterpret_tensor(buf229, (520, 1024), (1024, 1), 0), buf240, buf241, buf242, buf243, buf248, buf249, reinterpret_tensor(buf250, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf250, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf250, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf253, buf254, buf255, buf256, buf257, reinterpret_tensor(buf252, (520, 1024), (1024, 1), 0), buf263, buf264, buf265, buf266, buf272, buf273, reinterpret_tensor(buf274, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf274, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf274, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf277, buf278, buf279, buf280, buf281, reinterpret_tensor(buf276, (520, 1024), (1024, 1), 0), buf287, buf288, buf289, buf290, buf295, buf296, reinterpret_tensor(buf297, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf297, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf297, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048), buf300, buf301, buf302, buf303, buf304, reinterpret_tensor(buf299, (520, 1024), (1024, 1), 0), buf310, buf311, buf312, buf313, buf319, buf320, reinterpret_tensor(primals_171, (1000, 1024), (1024, 1), 0), buf322, reinterpret_tensor(primals_167, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_165, (4096, 1024), (1024, 1), 0), buf323, reinterpret_tensor(primals_161, (1024, 1024), (1024, 1), 0), buf324, reinterpret_tensor(primals_159, (3072, 1024), (1024, 1), 0), buf325, reinterpret_tensor(primals_155, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_153, (4096, 1024), (1024, 1), 0), buf326, reinterpret_tensor(primals_149, (1024, 1024), (1024, 1), 0), buf327, reinterpret_tensor(primals_147, (3072, 1024), (1024, 1), 0), buf328, reinterpret_tensor(primals_143, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_141, (4096, 1024), (1024, 1), 0), buf329, reinterpret_tensor(primals_137, (1024, 1024), (1024, 1), 0), buf330, reinterpret_tensor(primals_135, (3072, 1024), (1024, 1), 0), buf331, reinterpret_tensor(primals_131, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_129, (4096, 1024), (1024, 1), 0), buf332, reinterpret_tensor(primals_125, (1024, 1024), (1024, 1), 0), buf333, reinterpret_tensor(primals_123, (3072, 1024), (1024, 1), 0), reinterpret_tensor(primals_119, (1024, 512), (512, 1), 0), reinterpret_tensor(primals_115, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_113, (2048, 512), (512, 1), 0), buf334, reinterpret_tensor(primals_109, (512, 512), (512, 1), 0), buf335, reinterpret_tensor(primals_107, (1536, 512), (512, 1), 0), buf336, reinterpret_tensor(primals_103, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_101, (2048, 512), (512, 1), 0), buf337, reinterpret_tensor(primals_97, (512, 512), (512, 1), 0), buf338, reinterpret_tensor(primals_95, (1536, 512), (512, 1), 0), buf339, reinterpret_tensor(primals_91, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_89, (2048, 512), (512, 1), 0), buf340, reinterpret_tensor(primals_85, (512, 512), (512, 1), 0), buf341, reinterpret_tensor(primals_83, (1536, 512), (512, 1), 0), buf342, reinterpret_tensor(primals_79, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_77, (2048, 512), (512, 1), 0), buf343, reinterpret_tensor(primals_73, (512, 512), (512, 1), 0), buf344, reinterpret_tensor(primals_71, (1536, 512), (512, 1), 0), buf345, reinterpret_tensor(primals_67, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_65, (2048, 512), (512, 1), 0), buf346, reinterpret_tensor(primals_61, (512, 512), (512, 1), 0), buf347, reinterpret_tensor(primals_59, (1536, 512), (512, 1), 0), buf348, reinterpret_tensor(primals_55, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_53, (2048, 512), (512, 1), 0), buf349, reinterpret_tensor(primals_49, (512, 512), (512, 1), 0), buf350, reinterpret_tensor(primals_47, (1536, 512), (512, 1), 0), reinterpret_tensor(primals_43, (512, 256), (256, 1), 0), reinterpret_tensor(primals_39, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_37, (1024, 256), (256, 1), 0), buf351, reinterpret_tensor(primals_33, (256, 256), (256, 1), 0), buf352, reinterpret_tensor(primals_31, (768, 256), (256, 1), 0), buf353, reinterpret_tensor(primals_27, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_25, (1024, 256), (256, 1), 0), buf354, reinterpret_tensor(primals_21, (256, 256), (256, 1), 0), buf355, reinterpret_tensor(primals_19, (768, 256), (256, 1), 0), buf356, reinterpret_tensor(primals_15, (256, 1024), (1024, 1), 0), reinterpret_tensor(primals_13, (1024, 256), (256, 1), 0), buf357, reinterpret_tensor(primals_9, (256, 256), (256, 1), 0), buf358, reinterpret_tensor(primals_7, (768, 256), (256, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 256, 31, 31), (246016, 961, 31, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((256, 3, 14, 14), (588, 196, 14, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pit_b_224', benchmark_compiled_module)
