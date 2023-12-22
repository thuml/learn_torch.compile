
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


cpp_fused_native_layer_norm_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (1048576L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (1024L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (1048576L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (1024L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp6 = tmp5.exp();
                        tmp6.store(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (1048576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr4 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr3 + static_cast<long>(x2 + (64L*x1) + (65536L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (65536L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                tmp6.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1024.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr2 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_2, (1024, ), (1, ))
    assert_size_stride(primals_3, (1024, 1024), (1024, 1))
    assert_size_stride(primals_4, (1024, ), (1, ))
    assert_size_stride(primals_5, (1024, 1024), (1024, 1))
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_7, (1024, 1024), (1024, 1))
    assert_size_stride(primals_8, (1024, ), (1, ))
    assert_size_stride(primals_9, (1024, 1024), (1024, 1))
    assert_size_stride(primals_10, (1024, ), (1, ))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (1024, 1024), (1024, 1))
    assert_size_stride(primals_14, (1024, ), (1, ))
    assert_size_stride(primals_15, (1024, 1024), (1024, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    assert_size_stride(primals_17, (1024, 1024), (1024, 1))
    assert_size_stride(primals_18, (1024, ), (1, ))
    assert_size_stride(primals_19, (1024, 1024), (1024, 1))
    assert_size_stride(primals_20, (1024, ), (1, ))
    assert_size_stride(primals_21, (1024, ), (1, ))
    assert_size_stride(primals_22, (1024, ), (1, ))
    assert_size_stride(primals_23, (4096, 1024), (1024, 1))
    assert_size_stride(primals_24, (4096, ), (1, ))
    assert_size_stride(primals_25, (1024, 4096), (4096, 1))
    assert_size_stride(primals_26, (1024, ), (1, ))
    assert_size_stride(primals_27, (1, 1024, 1024), (1048576, 1024, 1))
    assert_size_stride(primals_28, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(primals_29, (1, 1024, 1024), (1048576, 1024, 1))
    buf0 = empty((1, 1024, 1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf3 = reinterpret_tensor(buf1, (1, 1024, 1), (1024, 1, 1), 0); del buf1  # reuse
    buf4 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_0(c_void_p(buf3.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf4.data_ptr()))
    del primals_2
    buf5 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_4, buf4, reinterpret_tensor(primals_3, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf5)
    del primals_4
    buf6 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, buf4, reinterpret_tensor(primals_5, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf6)
    del primals_6
    buf7 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf4, reinterpret_tensor(primals_7, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf7)
    del primals_8
    buf8 = empty((1, 16, 1024, 64), device='cpu', dtype=torch.float32)
    buf9 = empty((1, 16, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    buf10 = empty((16, 1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf8, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf9, (16, 64, 1024), (65536, 1, 64), 0), out=buf10)
    buf11 = empty_strided((16, 1024, 1), (1024, 1, 16384), device='cpu', dtype=torch.float32)
    buf12 = buf10; del buf10  # reuse
    buf13 = empty_strided((16, 1024, 1), (1024, 1, 16384), device='cpu', dtype=torch.float32)
    buf14 = empty((16, 1024, 1024), device='cpu', dtype=torch.float32)
    buf58 = empty((16, 1024, 1024), device='cpu', dtype=torch.float32)
    buf15 = reinterpret_tensor(buf6, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf6  # reuse
    cpp_fused__softmax_clone_detach_2(c_void_p(buf12.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_28
    buf16 = reinterpret_tensor(buf7, (16, 1024, 64), (65536, 64, 1), 0); del buf7  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf14, reinterpret_tensor(buf15, (16, 1024, 64), (65536, 64, 1), 0), out=buf16)
    buf17 = buf5; del buf5  # reuse
    cpp_fused_view_3(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = reinterpret_tensor(buf16, (1024, 1024), (1024, 1), 0); del buf16  # reuse
    # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, buf17, reinterpret_tensor(primals_9, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf18)
    del primals_10
    # Source Nodes: [hidden_states_2], Original ATen: [aten.native_dropout]
    buf19 = aten.native_dropout(reinterpret_tensor(buf18, (1, 1024, 1024), (1048576, 1024, 1), 0), 0.1, True)
    buf20 = buf19[0]
    buf21 = buf19[1]
    del buf19
    buf22 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf25 = reinterpret_tensor(buf18, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf18  # reuse
    buf26 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_4(c_void_p(primals_27.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del primals_12
    buf27 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf26, reinterpret_tensor(primals_13, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf27)
    del primals_14
    buf28 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, reinterpret_tensor(primals_29, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_15, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf28)
    del primals_16
    buf29 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, reinterpret_tensor(primals_29, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_17, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf29)
    del primals_18
    buf30 = empty((1, 16, 1024, 64), device='cpu', dtype=torch.float32)
    buf31 = empty((1, 16, 1024, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_5(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    buf32 = buf12; del buf12  # reuse
    # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf30, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf31, (16, 64, 1024), (65536, 1, 64), 0), out=buf32)
    buf33 = reinterpret_tensor(buf13, (16, 1024, 1), (1024, 1, 1), 0); del buf13  # reuse
    buf34 = empty((16, 1024, 1024), device='cpu', dtype=torch.float32)
    buf35 = reinterpret_tensor(buf11, (16, 1024, 1), (1024, 1, 1), 0); del buf11  # reuse
    buf36 = buf34; del buf34  # reuse
    buf37 = reinterpret_tensor(buf28, (1, 16, 1024, 64), (1048576, 65536, 64, 1), 0); del buf28  # reuse
    cpp_fused__softmax_clone_6(c_void_p(buf36.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = reinterpret_tensor(buf29, (16, 1024, 64), (65536, 64, 1), 0); del buf29  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf36, reinterpret_tensor(buf37, (16, 1024, 64), (65536, 64, 1), 0), out=buf38)
    buf39 = buf27; del buf27  # reuse
    cpp_fused_view_7(c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = reinterpret_tensor(buf38, (1024, 1024), (1024, 1), 0); del buf38  # reuse
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_20, buf39, reinterpret_tensor(primals_19, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf40)
    del primals_20
    # Source Nodes: [hidden_states_6], Original ATen: [aten.native_dropout]
    buf41 = aten.native_dropout(reinterpret_tensor(buf40, (1, 1024, 1024), (1048576, 1024, 1), 0), 0.1, True)
    buf42 = buf41[0]
    buf43 = buf41[1]
    del buf41
    buf44 = buf22; del buf22  # reuse
    buf45 = empty_strided((1, 1024, 1), (1024, 1, 1024), device='cpu', dtype=torch.float32)
    buf47 = reinterpret_tensor(buf40, (1, 1024, 1024), (1048576, 1024, 1), 0); del buf40  # reuse
    buf48 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_8(c_void_p(primals_27.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del buf44
    del primals_22
    buf49 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf48, reinterpret_tensor(primals_23, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf49)
    del primals_24
    buf50 = empty((1024, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_9(c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = empty((1024, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf50, reinterpret_tensor(primals_25, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf51)
    del primals_26
    # Source Nodes: [hidden_states_12], Original ATen: [aten.native_dropout]
    buf52 = aten.native_dropout(reinterpret_tensor(buf51, (1, 1024, 1024), (1048576, 1024, 1), 0), 0.1, True)
    del buf51
    buf53 = buf52[0]
    buf54 = buf52[1]
    del buf52
    buf55 = buf53; del buf53  # reuse
    buf56 = reinterpret_tensor(buf45, (1, 1024, 1), (1024, 1, 1), 0); del buf45  # reuse
    buf57 = reinterpret_tensor(buf23, (1, 1024, 1), (1024, 1, 1), 0); del buf23  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_10(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf42.data_ptr()))
    return (buf55, primals_1, primals_11, primals_21, primals_27, buf0, buf3, buf4, buf17, buf21, buf25, buf26, reinterpret_tensor(primals_29, (1024, 1024), (1024, 1), 0), buf32, buf33, buf35, buf39, buf43, buf47, buf48, buf49, buf50, buf54, reinterpret_tensor(primals_25, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_23, (4096, 1024), (1024, 1), 0), buf56, reinterpret_tensor(primals_19, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf36, (16, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf37, (16, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf30, (16, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf31, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(primals_17, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_15, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_13, (1024, 1024), (1024, 1), 0), buf57, reinterpret_tensor(primals_9, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf14, (16, 1024, 1024), (1048576, 1, 1024), 0), reinterpret_tensor(buf15, (16, 64, 1024), (65536, 1, 64), 0), buf58, reinterpret_tensor(buf8, (16, 64, 1024), (65536, 1, 64), 0), reinterpret_tensor(buf9, (16, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(primals_7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_5, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 1024), (1024, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((1, 1024, 1024), (1048576, 1024, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('MBartForConditionalGeneration', benchmark_compiled_module)
