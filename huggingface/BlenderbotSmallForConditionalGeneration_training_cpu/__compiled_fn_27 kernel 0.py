
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


cpp_fused_clone_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_1 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (128L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp6 = tmp5.exp();
                        tmp6.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr4 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*x0) + (4096L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_3 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                auto tmp9 = static_cast<float>(1e-05);
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
''')


cpp_fused_clone_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    tmp3.store(out_ptr0 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                    tmp0.store(out_ptr1 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp4.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (32L*x0) + (512L*x1)));
                            tmp0.store(out_ptr3 + static_cast<long>(x2 + (32L*x1) + (4096L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*x0) + (4096L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
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
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = tmp0 * tmp1;
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (512L*x0)));
                auto tmp7 = out_ptr0[static_cast<long>(x0)];
                auto tmp10 = out_ptr1[static_cast<long>(x0)];
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp2 = tmp0 * tmp1;
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
                tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                tmp21.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            auto tmp8 = tmp7 / tmp2;
            tmp8.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(512.0);
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
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29 = args
    args.clear()
    assert_size_stride(primals_1, (512, 512), (512, 1))
    assert_size_stride(primals_2, (512, ), (1, ))
    assert_size_stride(primals_3, (512, 512), (512, 1))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, 512), (512, 1))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, 512), (512, 1))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, 512), (512, 1))
    assert_size_stride(primals_12, (512, ), (1, ))
    assert_size_stride(primals_13, (512, 512), (512, 1))
    assert_size_stride(primals_14, (512, ), (1, ))
    assert_size_stride(primals_15, (512, 512), (512, 1))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, 512), (512, 1))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (2048, 512), (512, 1))
    assert_size_stride(primals_22, (2048, ), (1, ))
    assert_size_stride(primals_23, (512, 2048), (2048, 1))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (1, 128, 512), (65536, 512, 1))
    assert_size_stride(primals_28, (1, 1, 128, 128), (16384, 16384, 128, 1))
    assert_size_stride(primals_29, (1, 128, 512), (65536, 512, 1))
    buf0 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_2, reinterpret_tensor(primals_27, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf0)
    del primals_2
    buf1 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_4, reinterpret_tensor(primals_27, (128, 512), (512, 1), 0), reinterpret_tensor(primals_3, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf1)
    del primals_4
    buf2 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, reinterpret_tensor(primals_27, (128, 512), (512, 1), 0), reinterpret_tensor(primals_5, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf2)
    del primals_6
    buf3 = empty((1, 16, 128, 32), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 16, 128, 32), device='cpu', dtype=torch.float32)
    cpp_fused_clone_0(c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    buf5 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf3, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf4, (16, 32, 128), (4096, 1, 32), 0), out=buf5)
    buf6 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf7 = buf5; del buf5  # reuse
    buf8 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf9 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf58 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf1, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf1  # reuse
    cpp_fused__softmax_clone_detach_1(c_void_p(buf7.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf10.data_ptr()))
    del primals_28
    buf11 = reinterpret_tensor(buf2, (16, 128, 32), (4096, 32, 1), 0); del buf2  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf9, reinterpret_tensor(buf10, (16, 128, 32), (4096, 32, 1), 0), out=buf11)
    buf12 = buf0; del buf0  # reuse
    cpp_fused_view_2(c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()))
    buf13 = reinterpret_tensor(buf11, (128, 512), (512, 1), 0); del buf11  # reuse
    # Source Nodes: [hidden_states], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf12, reinterpret_tensor(primals_7, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf13)
    del primals_8
    # Source Nodes: [hidden_states_1], Original ATen: [aten.native_dropout]
    buf14 = aten.native_dropout(reinterpret_tensor(buf13, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf20 = reinterpret_tensor(buf13, (1, 128, 512), (65536, 512, 1), 0); del buf13  # reuse
    buf21 = empty((128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_3(c_void_p(primals_27.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = reinterpret_tensor(buf15, (128, 512), (512, 1), 0); del buf15  # reuse
    # Source Nodes: [l__self___encoder_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_12, buf21, reinterpret_tensor(primals_11, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf22)
    del primals_12
    buf23 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___encoder_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, reinterpret_tensor(primals_29, (128, 512), (512, 1), 0), reinterpret_tensor(primals_13, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf23)
    del primals_14
    buf24 = empty((128, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___encoder_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, reinterpret_tensor(primals_29, (128, 512), (512, 1), 0), reinterpret_tensor(primals_15, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf24)
    del primals_16
    buf25 = empty((1, 16, 128, 32), device='cpu', dtype=torch.float32)
    buf26 = empty((1, 16, 128, 32), device='cpu', dtype=torch.float32)
    cpp_fused_clone_4(c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    buf27 = buf7; del buf7  # reuse
    # Source Nodes: [attn_weights_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf26, (16, 32, 128), (4096, 1, 32), 0), out=buf27)
    buf28 = reinterpret_tensor(buf8, (16, 128, 1), (128, 1, 1), 0); del buf8  # reuse
    buf29 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf30 = reinterpret_tensor(buf6, (16, 128, 1), (128, 1, 1), 0); del buf6  # reuse
    buf31 = buf29; del buf29  # reuse
    buf32 = reinterpret_tensor(buf23, (1, 16, 128, 32), (65536, 4096, 32, 1), 0); del buf23  # reuse
    cpp_fused__softmax_clone_5(c_void_p(buf31.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf24, (16, 128, 32), (4096, 32, 1), 0); del buf24  # reuse
    # Source Nodes: [attn_output_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf31, reinterpret_tensor(buf32, (16, 128, 32), (4096, 32, 1), 0), out=buf33)
    buf34 = buf22; del buf22  # reuse
    cpp_fused_view_6(c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()))
    buf35 = reinterpret_tensor(buf33, (128, 512), (512, 1), 0); del buf33  # reuse
    # Source Nodes: [hidden_states_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, buf34, reinterpret_tensor(primals_17, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf35)
    del primals_18
    # Source Nodes: [hidden_states_5], Original ATen: [aten.native_dropout]
    buf36 = aten.native_dropout(reinterpret_tensor(buf35, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = buf17; del buf17  # reuse
    buf40 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf42 = reinterpret_tensor(buf35, (1, 128, 512), (65536, 512, 1), 0); del buf35  # reuse
    buf43 = empty((128, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf20.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del primals_10
    buf44 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, buf43, reinterpret_tensor(primals_21, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf44)
    del primals_22
    buf45 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_8(c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()))
    buf46 = reinterpret_tensor(buf37, (128, 512), (512, 1), 0); del buf37  # reuse
    # Source Nodes: [hidden_states_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf45, reinterpret_tensor(primals_23, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf46)
    del primals_24
    # Source Nodes: [hidden_states_11], Original ATen: [aten.native_dropout]
    buf47 = aten.native_dropout(reinterpret_tensor(buf46, (1, 128, 512), (65536, 512, 1), 0), 0.1, True)
    buf48 = buf47[0]
    buf49 = buf47[1]
    del buf47
    buf50 = buf39; del buf39  # reuse
    buf51 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf53 = reinterpret_tensor(buf46, (1, 128, 512), (65536, 512, 1), 0); del buf46  # reuse
    buf54 = empty((1, 128, 512), device='cpu', dtype=torch.float32)
    buf55 = reinterpret_tensor(buf51, (1, 128, 1), (128, 1, 1), 0); del buf51  # reuse
    buf56 = reinterpret_tensor(buf40, (1, 128, 1), (128, 1, 1), 0); del buf40  # reuse
    buf57 = reinterpret_tensor(buf18, (1, 128, 1), (128, 1, 1), 0); del buf18  # reuse
    cpp_fused_add_native_layer_norm_native_layer_norm_backward_9(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del buf48
    del buf50
    del primals_20
    del primals_26
    return (buf54, primals_9, primals_19, primals_25, reinterpret_tensor(primals_27, (128, 512), (512, 1), 0), buf12, buf16, buf20, buf21, reinterpret_tensor(primals_29, (128, 512), (512, 1), 0), buf27, buf28, buf30, buf34, buf38, buf42, buf43, buf44, buf45, buf49, buf53, buf55, reinterpret_tensor(primals_23, (512, 2048), (2048, 1), 0), reinterpret_tensor(primals_21, (2048, 512), (512, 1), 0), buf56, reinterpret_tensor(primals_17, (512, 512), (512, 1), 0), reinterpret_tensor(buf31, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf32, (16, 32, 128), (4096, 1, 32), 0), reinterpret_tensor(buf25, (16, 32, 128), (4096, 1, 32), 0), reinterpret_tensor(buf26, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(primals_15, (512, 512), (512, 1), 0), reinterpret_tensor(primals_13, (512, 512), (512, 1), 0), reinterpret_tensor(primals_11, (512, 512), (512, 1), 0), buf57, reinterpret_tensor(primals_7, (512, 512), (512, 1), 0), reinterpret_tensor(buf9, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf10, (16, 32, 128), (4096, 1, 32), 0), buf58, reinterpret_tensor(buf3, (16, 32, 128), (4096, 1, 32), 0), reinterpret_tensor(buf4, (16, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(primals_5, (512, 512), (512, 1), 0), reinterpret_tensor(primals_3, (512, 512), (512, 1), 0), reinterpret_tensor(primals_1, (512, 512), (512, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((1, 128, 512), (65536, 512, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BlenderbotSmallForConditionalGeneration', benchmark_compiled_module)
