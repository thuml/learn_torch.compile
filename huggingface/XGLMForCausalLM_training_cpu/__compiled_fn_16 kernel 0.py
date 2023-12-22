
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x0) + (1024L*x1)));
                        auto tmp1 = static_cast<float>(0.125);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (8192L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
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
                            auto tmp3 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp4 = at::vec::Vectorized<float>(tmp3);
                            auto tmp5 = at::vec::maximum(tmp2, tmp4);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1)));
                        auto tmp6 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = at::vec::maximum(tmp2, tmp4);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*x0) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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


cpp_fused_gelu_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
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


cpp_fused_add_eq_lift_fresh_lt_native_layer_norm_native_layer_norm_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       bool* out_ptr0,
                       bool* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16384L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr2[static_cast<long>(x1 + (16384L*x0))];
                    auto tmp1 = in_ptr3[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp3 = static_cast<float>(-3.4028234663852886e+38);
                    auto tmp4 = tmp2 == tmp3;
                    auto tmp5 = tmp2 < tmp3;
                    out_ptr0[static_cast<long>(x1 + (16384L*x0))] = tmp4;
                    out_ptr1[static_cast<long>(x1 + (16384L*x0))] = tmp5;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18 = args
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
    assert_size_stride(primals_13, (4096, 1024), (1024, 1))
    assert_size_stride(primals_14, (4096, ), (1, ))
    assert_size_stride(primals_15, (1024, 4096), (4096, 1))
    assert_size_stride(primals_16, (1024, ), (1, ))
    assert_size_stride(primals_17, (1, 128, 1024), (131072, 1024, 1))
    assert_size_stride(primals_18, (1, 1, 128, 128), (16384, 16384, 128, 1))
    buf0 = empty((1, 128, 1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = reinterpret_tensor(buf1, (1, 128, 1), (128, 1, 1), 0); del buf1  # reuse
    buf4 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_0(c_void_p(buf3.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf4.data_ptr()))
    del primals_2
    buf5 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_q_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_4, buf4, reinterpret_tensor(primals_3, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf5)
    del primals_4
    buf6 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___self_attn_k_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_6, buf4, reinterpret_tensor(primals_5, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf6)
    del primals_6
    buf7 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    buf8 = buf6; del buf6  # reuse
    # Source Nodes: [l__self___self_attn_v_proj], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_8, buf4, reinterpret_tensor(primals_7, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf8)
    del primals_8
    buf9 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    buf10 = empty((1, 16, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_2(c_void_p(buf8.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    buf11 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf10, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf7, (16, 64, 128), (8192, 1, 64), 0), out=buf11)
    buf12 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf13 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((16, 128, 1), (128, 1, 2048), device='cpu', dtype=torch.float32)
    buf15 = buf13; del buf13  # reuse
    cpp_fused__softmax_3(c_void_p(buf15.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()))
    del buf12
    del buf14
    # Source Nodes: [attn_probs, attn_weights_4], Original ATen: [aten._softmax, aten.native_dropout]
    buf16 = aten.native_dropout(buf15, 0.1, True)
    buf17 = buf16[0]
    buf18 = buf16[1]
    del buf16
    buf19 = reinterpret_tensor(buf8, (16, 128, 64), (8192, 64, 1), 0); del buf8  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(buf17, reinterpret_tensor(buf9, (16, 128, 64), (8192, 64, 1), 0), out=buf19)
    buf20 = buf5; del buf5  # reuse
    cpp_fused_view_4(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()))
    buf21 = reinterpret_tensor(buf19, (128, 1024), (1024, 1), 0); del buf19  # reuse
    # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_10, buf20, reinterpret_tensor(primals_9, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf21)
    del primals_10
    # Source Nodes: [hidden_states_2], Original ATen: [aten.native_dropout]
    buf22 = aten.native_dropout(reinterpret_tensor(buf21, (1, 128, 1024), (131072, 1024, 1), 0), 0.1, True)
    buf23 = buf22[0]
    buf24 = buf22[1]
    del buf22
    buf25 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf28 = reinterpret_tensor(buf21, (1, 128, 1024), (131072, 1024, 1), 0); del buf21  # reuse
    buf29 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_5(c_void_p(primals_17.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del buf25
    del primals_12
    buf30 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__self___fc1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_14, buf29, reinterpret_tensor(primals_13, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf30)
    del primals_14
    buf31 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_6(c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    buf32 = empty((128, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_16, buf31, reinterpret_tensor(primals_15, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf32)
    del primals_16
    # Source Nodes: [hidden_states_8], Original ATen: [aten.native_dropout]
    buf33 = aten.native_dropout(reinterpret_tensor(buf32, (1, 128, 1024), (131072, 1024, 1), 0), 0.1, True)
    del buf32
    buf34 = buf33[0]
    buf35 = buf33[1]
    del buf33
    buf36 = buf34; del buf34  # reuse
    buf37 = reinterpret_tensor(buf26, (1, 128, 1), (128, 1, 1), 0); del buf26  # reuse
    buf38 = empty((1, 16, 128, 128), device='cpu', dtype=torch.bool)
    buf39 = empty((1, 16, 128, 128), device='cpu', dtype=torch.bool)
    cpp_fused_add_eq_lift_fresh_lt_native_layer_norm_native_layer_norm_backward_7(c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    del buf11
    del buf23
    del primals_18
    return (buf36, buf7, buf9, primals_1, primals_11, primals_17, buf0, buf3, buf4, buf18, buf20, buf24, buf28, buf29, buf30, buf31, buf35, reinterpret_tensor(primals_15, (1024, 4096), (4096, 1), 0), reinterpret_tensor(primals_13, (4096, 1024), (1024, 1), 0), buf37, reinterpret_tensor(primals_9, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf17, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf9, (16, 64, 128), (8192, 1, 64), 0), buf15, buf38, buf39, reinterpret_tensor(buf10, (16, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf7, (16, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(primals_7, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_5, (1024, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 1024), (1024, 1), 0), )


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
    primals_13 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((1, 128, 1024), (131072, 1024, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((1, 1, 128, 128), (16384, 16384, 128, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('XGLMForCausalLM', benchmark_compiled_module)
