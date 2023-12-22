
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


cpp_fused_add_embedding_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1)));
                            auto tmp1 = decltype(tmp0)(tmp0 + 50257);
                            auto tmp2 = tmp0 < 0;
                            auto tmp3 = tmp2 ? tmp1 : tmp0;
                            TORCH_CHECK((0 <= tmp3) & (tmp3 < 50257L), "index out of bounds: 0 <= tmp3 < 50257L")
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*tmp3)));
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50257);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50257L), "index out of bounds: 0 <= tmp3 < 50257L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(768.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_native_layer_norm_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const long* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (196608L*x0)));
                            auto tmp1 = in_ptr1[static_cast<long>(x1 + (256L*x0))];
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1)));
                            auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                            auto tmp3 = tmp1 < 0;
                            auto tmp4 = tmp3 ? tmp2 : tmp1;
                            TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*tmp4)));
                            auto tmp7 = tmp5 + tmp6;
                            auto tmp8 = tmp0 + tmp7;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (196608L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp9 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp12 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                        auto tmp3 = tmp1 < 0;
                        auto tmp4 = tmp3 ? tmp2 : tmp1;
                        TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*tmp4)));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp0 + tmp7;
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 - tmp10;
                        auto tmp13 = static_cast<float>(768.0);
                        auto tmp14 = tmp12 / tmp13;
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                        auto tmp17 = 1 / std::sqrt(tmp16);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = tmp11 * tmp18;
                        auto tmp21 = tmp19 * tmp20;
                        auto tmp23 = tmp21 + tmp22;
                        tmp23.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (196608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_native_layer_norm_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (196608L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x1 + (256L*x0))];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (196608L*x0)));
                        auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                        auto tmp3 = tmp1 < 0;
                        auto tmp4 = tmp3 ? tmp2 : tmp1;
                        TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*tmp4)));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp0 + tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        tmp10.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (196608L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp0 + tmp5;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = tmp1 + tmp2;
                auto tmp5 = tmp3 + tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (589824L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_div_full_where_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                            auto tmp2 = static_cast<float>(8.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (1024L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                        auto tmp2 = static_cast<float>(8.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(1536L + x3 + (64L*x1) + (2304L*x2) + (589824L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (16384L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (16384L*x2) + (196608L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (196608L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.5);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 * tmp2;
                auto tmp4 = tmp0 * tmp0;
                auto tmp5 = tmp4 * tmp0;
                auto tmp6 = static_cast<float>(0.044715);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 * tmp7;
                auto tmp9 = tmp0 + tmp8;
                auto tmp10 = static_cast<float>(0.7978845608028654);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = decltype(tmp12)(2) / (decltype(tmp12)(1) + (decltype(tmp12)(-2) * tmp12).exp()) - decltype(tmp12)(1);
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp13 + tmp15;
                auto tmp17 = tmp3 * tmp16;
                tmp17.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2304, ), (1, ))
    assert_size_stride(arg1_1, (768, 2304), (2304, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, 768), (768, 1))
    assert_size_stride(arg4_1, (3072, ), (1, ))
    assert_size_stride(arg5_1, (768, 3072), (3072, 1))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (3072, 768), (768, 1))
    assert_size_stride(arg8_1, (2304, ), (1, ))
    assert_size_stride(arg9_1, (768, 2304), (2304, 1))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, 768), (768, 1))
    assert_size_stride(arg12_1, (3072, ), (1, ))
    assert_size_stride(arg13_1, (768, 3072), (3072, 1))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (3072, 768), (768, 1))
    assert_size_stride(arg16_1, (2304, ), (1, ))
    assert_size_stride(arg17_1, (768, 2304), (2304, 1))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, 768), (768, 1))
    assert_size_stride(arg20_1, (3072, ), (1, ))
    assert_size_stride(arg21_1, (768, 3072), (3072, 1))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (3072, 768), (768, 1))
    assert_size_stride(arg24_1, (2304, ), (1, ))
    assert_size_stride(arg25_1, (768, 2304), (2304, 1))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, 768), (768, 1))
    assert_size_stride(arg28_1, (3072, ), (1, ))
    assert_size_stride(arg29_1, (768, 3072), (3072, 1))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (3072, 768), (768, 1))
    assert_size_stride(arg32_1, (2304, ), (1, ))
    assert_size_stride(arg33_1, (768, 2304), (2304, 1))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, 768), (768, 1))
    assert_size_stride(arg36_1, (3072, ), (1, ))
    assert_size_stride(arg37_1, (768, 3072), (3072, 1))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (3072, 768), (768, 1))
    assert_size_stride(arg40_1, (2304, ), (1, ))
    assert_size_stride(arg41_1, (768, 2304), (2304, 1))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, 768), (768, 1))
    assert_size_stride(arg44_1, (3072, ), (1, ))
    assert_size_stride(arg45_1, (768, 3072), (3072, 1))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (3072, 768), (768, 1))
    assert_size_stride(arg48_1, (2304, ), (1, ))
    assert_size_stride(arg49_1, (768, 2304), (2304, 1))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, 768), (768, 1))
    assert_size_stride(arg52_1, (3072, ), (1, ))
    assert_size_stride(arg53_1, (768, 3072), (3072, 1))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (3072, 768), (768, 1))
    assert_size_stride(arg56_1, (2304, ), (1, ))
    assert_size_stride(arg57_1, (768, 2304), (2304, 1))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, 768), (768, 1))
    assert_size_stride(arg60_1, (3072, ), (1, ))
    assert_size_stride(arg61_1, (768, 3072), (3072, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (3072, 768), (768, 1))
    assert_size_stride(arg64_1, (2304, ), (1, ))
    assert_size_stride(arg65_1, (768, 2304), (2304, 1))
    assert_size_stride(arg66_1, (768, ), (1, ))
    assert_size_stride(arg67_1, (768, 768), (768, 1))
    assert_size_stride(arg68_1, (3072, ), (1, ))
    assert_size_stride(arg69_1, (768, 3072), (3072, 1))
    assert_size_stride(arg70_1, (768, ), (1, ))
    assert_size_stride(arg71_1, (3072, 768), (768, 1))
    assert_size_stride(arg72_1, (2304, ), (1, ))
    assert_size_stride(arg73_1, (768, 2304), (2304, 1))
    assert_size_stride(arg74_1, (768, ), (1, ))
    assert_size_stride(arg75_1, (768, 768), (768, 1))
    assert_size_stride(arg76_1, (3072, ), (1, ))
    assert_size_stride(arg77_1, (768, 3072), (3072, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (3072, 768), (768, 1))
    assert_size_stride(arg80_1, (2304, ), (1, ))
    assert_size_stride(arg81_1, (768, 2304), (2304, 1))
    assert_size_stride(arg82_1, (768, ), (1, ))
    assert_size_stride(arg83_1, (768, 768), (768, 1))
    assert_size_stride(arg84_1, (3072, ), (1, ))
    assert_size_stride(arg85_1, (768, 3072), (3072, 1))
    assert_size_stride(arg86_1, (768, ), (1, ))
    assert_size_stride(arg87_1, (3072, 768), (768, 1))
    assert_size_stride(arg88_1, (2304, ), (1, ))
    assert_size_stride(arg89_1, (768, 2304), (2304, 1))
    assert_size_stride(arg90_1, (768, ), (1, ))
    assert_size_stride(arg91_1, (768, 768), (768, 1))
    assert_size_stride(arg92_1, (3072, ), (1, ))
    assert_size_stride(arg93_1, (768, 3072), (3072, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (3072, 768), (768, 1))
    assert_size_stride(arg96_1, (50257, 768), (768, 1))
    assert_size_stride(arg97_1, (1024, 768), (768, 1))
    assert_size_stride(arg98_1, (768, ), (1, ))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, ), (1, ))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, ), (1, ))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (768, ), (1, ))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (768, ), (1, ))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, ), (1, ))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, ), (1, ))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, ), (1, ))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, ), (1, ))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, ), (1, ))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, ), (1, ))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, ), (1, ))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, ), (1, ))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, ), (1, ))
    assert_size_stride(arg132_1, (768, ), (1, ))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, ), (1, ))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, ), (1, ))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, ), (1, ))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, ), (1, ))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (768, ), (1, ))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, ), (1, ))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (50257, 768), (768, 1))
    assert_size_stride(arg149_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg150_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg151_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg152_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg153_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg154_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg155_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg156_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg157_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg158_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg159_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg160_1, (1, 1, 1024, 1024), (1048576, 1048576, 1024, 1))
    assert_size_stride(arg161_1, (2, 256), (256, 1))
    buf0 = empty_strided((2, 256, 1), (256, 1, 512), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((2, 256, 1), (256, 1, 512), device='cpu', dtype=torch.float32)
    buf3 = empty((2, 256, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg161_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg98_1
    del arg99_1
    buf4 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg0_1, reinterpret_tensor(buf3, (512, 768), (768, 1), 0), arg1_1, alpha=1, beta=1, out=buf4)
    del arg0_1
    del arg1_1
    buf5 = reinterpret_tensor(buf3, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf3  # reuse
    buf6 = empty((2, 12, 64, 256), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    buf7 = empty((24, 256, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf5, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf6, (24, 64, 256), (16384, 256, 1), 0), out=buf7)
    buf8 = empty_strided((2, 12, 256, 1), (3072, 256, 1, 6144), device='cpu', dtype=torch.float32)
    buf9 = reinterpret_tensor(buf7, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf7  # reuse
    buf10 = empty_strided((2, 12, 256, 1), (3072, 256, 1, 6144), device='cpu', dtype=torch.float32)
    buf11 = buf9; del buf9  # reuse
    buf12 = reinterpret_tensor(buf6, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf6  # reuse
    cpp_fused__softmax_clone_div_full_where_2(c_void_p(buf11.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg149_1
    buf13 = reinterpret_tensor(buf5, (24, 256, 64), (16384, 64, 1), 0); del buf5  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf11, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf12, (24, 256, 64), (16384, 64, 1), 0), out=buf13)
    buf14 = reinterpret_tensor(buf12, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf12  # reuse
    cpp_fused_clone_3(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    buf15 = reinterpret_tensor(buf13, (512, 768), (768, 1), 0); del buf13  # reuse
    # Source Nodes: [x_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg2_1, reinterpret_tensor(buf14, (512, 768), (768, 1), 0), arg3_1, alpha=1, beta=1, out=buf15)
    del arg2_1
    del arg3_1
    buf16 = buf1; del buf1  # reuse
    buf17 = buf0; del buf0  # reuse
    buf19 = reinterpret_tensor(buf14, (2, 256, 768), (196608, 768, 1), 0); del buf14  # reuse
    cpp_fused_add_embedding_native_layer_norm_4(c_void_p(buf15.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()))
    del arg100_1
    del arg101_1
    buf20 = reinterpret_tensor(buf11, (512, 3072), (3072, 1), 0); del buf11  # reuse
    # Source Nodes: [x_4], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg4_1, reinterpret_tensor(buf19, (512, 768), (768, 1), 0), arg5_1, alpha=1, beta=1, out=buf20)
    del arg4_1
    del arg5_1
    buf21 = reinterpret_tensor(buf20, (2, 256, 3072), (786432, 3072, 1), 0); del buf20  # reuse
    cpp_fused_add_mul_pow_tanh_5(c_void_p(buf21.data_ptr()))
    buf22 = reinterpret_tensor(buf19, (512, 768), (768, 1), 0); del buf19  # reuse
    # Source Nodes: [x_6], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg6_1, reinterpret_tensor(buf21, (512, 3072), (3072, 1), 0), arg7_1, alpha=1, beta=1, out=buf22)
    del arg6_1
    del arg7_1
    buf23 = reinterpret_tensor(buf22, (2, 256, 768), (196608, 768, 1), 0); del buf22  # reuse
    buf24 = buf17; del buf17  # reuse
    buf25 = buf16; del buf16  # reuse
    buf27 = empty((2, 256, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_6(c_void_p(buf23.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf27.data_ptr()))
    del arg102_1
    del arg103_1
    del arg161_1
    del arg96_1
    del arg97_1
    buf28 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf27, (512, 768), (768, 1), 0), arg9_1, alpha=1, beta=1, out=buf28)
    del arg8_1
    del arg9_1
    buf29 = reinterpret_tensor(buf27, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf27  # reuse
    buf30 = reinterpret_tensor(buf15, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf15  # reuse
    cpp_fused_clone_7(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    buf31 = reinterpret_tensor(buf21, (24, 256, 256), (65536, 256, 1), 0); del buf21  # reuse
    # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf29, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf30, (24, 64, 256), (16384, 256, 1), 0), out=buf31)
    buf32 = buf8; del buf8  # reuse
    buf33 = reinterpret_tensor(buf31, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf31  # reuse
    buf34 = buf10; del buf10  # reuse
    buf35 = buf33; del buf33  # reuse
    buf36 = reinterpret_tensor(buf30, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf30  # reuse
    cpp_fused__softmax_clone_div_full_where_8(c_void_p(buf35.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()))
    del arg150_1
    buf37 = reinterpret_tensor(buf29, (24, 256, 64), (16384, 64, 1), 0); del buf29  # reuse
    # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf35, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf36, (24, 256, 64), (16384, 64, 1), 0), out=buf37)
    buf38 = reinterpret_tensor(buf36, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf36  # reuse
    cpp_fused_clone_9(c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf37, (512, 768), (768, 1), 0); del buf37  # reuse
    # Source Nodes: [x_10], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf38, (512, 768), (768, 1), 0), arg11_1, alpha=1, beta=1, out=buf39)
    del arg10_1
    del arg11_1
    buf40 = buf25; del buf25  # reuse
    buf41 = buf24; del buf24  # reuse
    buf43 = reinterpret_tensor(buf38, (2, 256, 768), (196608, 768, 1), 0); del buf38  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf39.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg104_1
    del arg105_1
    buf44 = reinterpret_tensor(buf35, (512, 3072), (3072, 1), 0); del buf35  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf43, (512, 768), (768, 1), 0), arg13_1, alpha=1, beta=1, out=buf44)
    del arg12_1
    del arg13_1
    buf45 = reinterpret_tensor(buf44, (2, 256, 3072), (786432, 3072, 1), 0); del buf44  # reuse
    cpp_fused_add_mul_pow_tanh_11(c_void_p(buf45.data_ptr()))
    buf46 = reinterpret_tensor(buf43, (512, 768), (768, 1), 0); del buf43  # reuse
    # Source Nodes: [x_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf45, (512, 3072), (3072, 1), 0), arg15_1, alpha=1, beta=1, out=buf46)
    del arg14_1
    del arg15_1
    buf47 = buf41; del buf41  # reuse
    buf48 = buf40; del buf40  # reuse
    buf50 = empty((2, 256, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_12(c_void_p(buf39.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg106_1
    del arg107_1
    buf51 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg16_1, reinterpret_tensor(buf50, (512, 768), (768, 1), 0), arg17_1, alpha=1, beta=1, out=buf51)
    del arg16_1
    del arg17_1
    buf52 = reinterpret_tensor(buf50, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf50  # reuse
    buf53 = empty((2, 12, 64, 256), device='cpu', dtype=torch.float32)
    cpp_fused_clone_13(c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    buf54 = reinterpret_tensor(buf45, (24, 256, 256), (65536, 256, 1), 0); del buf45  # reuse
    # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf52, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf53, (24, 64, 256), (16384, 256, 1), 0), out=buf54)
    buf55 = buf34; del buf34  # reuse
    buf56 = reinterpret_tensor(buf54, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf54  # reuse
    buf57 = buf32; del buf32  # reuse
    buf58 = buf56; del buf56  # reuse
    buf59 = reinterpret_tensor(buf53, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf53  # reuse
    cpp_fused__softmax_clone_div_full_where_14(c_void_p(buf58.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf59.data_ptr()))
    del arg151_1
    buf60 = reinterpret_tensor(buf52, (24, 256, 64), (16384, 64, 1), 0); del buf52  # reuse
    # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf58, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf59, (24, 256, 64), (16384, 64, 1), 0), out=buf60)
    buf61 = reinterpret_tensor(buf59, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf59  # reuse
    cpp_fused_clone_15(c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = reinterpret_tensor(buf60, (512, 768), (768, 1), 0); del buf60  # reuse
    # Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf61, (512, 768), (768, 1), 0), arg19_1, alpha=1, beta=1, out=buf62)
    del arg18_1
    del arg19_1
    buf63 = buf48; del buf48  # reuse
    buf64 = buf47; del buf47  # reuse
    buf66 = reinterpret_tensor(buf61, (2, 256, 768), (196608, 768, 1), 0); del buf61  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf62.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()))
    del arg108_1
    del arg109_1
    buf67 = reinterpret_tensor(buf58, (512, 3072), (3072, 1), 0); del buf58  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf66, (512, 768), (768, 1), 0), arg21_1, alpha=1, beta=1, out=buf67)
    del arg20_1
    del arg21_1
    buf68 = reinterpret_tensor(buf67, (2, 256, 3072), (786432, 3072, 1), 0); del buf67  # reuse
    cpp_fused_add_mul_pow_tanh_17(c_void_p(buf68.data_ptr()))
    buf69 = reinterpret_tensor(buf66, (512, 768), (768, 1), 0); del buf66  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg22_1, reinterpret_tensor(buf68, (512, 3072), (3072, 1), 0), arg23_1, alpha=1, beta=1, out=buf69)
    del arg22_1
    del arg23_1
    buf70 = reinterpret_tensor(buf69, (2, 256, 768), (196608, 768, 1), 0); del buf69  # reuse
    buf71 = buf64; del buf64  # reuse
    buf72 = buf63; del buf63  # reuse
    buf74 = empty((2, 256, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_18(c_void_p(buf70.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg110_1
    del arg111_1
    buf75 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg24_1, reinterpret_tensor(buf74, (512, 768), (768, 1), 0), arg25_1, alpha=1, beta=1, out=buf75)
    del arg24_1
    del arg25_1
    buf76 = reinterpret_tensor(buf74, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf74  # reuse
    buf77 = reinterpret_tensor(buf62, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf62  # reuse
    cpp_fused_clone_19(c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()))
    buf78 = reinterpret_tensor(buf68, (24, 256, 256), (65536, 256, 1), 0); del buf68  # reuse
    # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf76, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf77, (24, 64, 256), (16384, 256, 1), 0), out=buf78)
    buf79 = buf57; del buf57  # reuse
    buf80 = reinterpret_tensor(buf78, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf78  # reuse
    buf81 = buf55; del buf55  # reuse
    buf82 = buf80; del buf80  # reuse
    buf83 = reinterpret_tensor(buf77, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf77  # reuse
    cpp_fused__softmax_clone_div_full_where_20(c_void_p(buf82.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg152_1
    buf84 = reinterpret_tensor(buf76, (24, 256, 64), (16384, 64, 1), 0); del buf76  # reuse
    # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf82, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf83, (24, 256, 64), (16384, 64, 1), 0), out=buf84)
    buf85 = reinterpret_tensor(buf83, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf83  # reuse
    cpp_fused_clone_21(c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = reinterpret_tensor(buf84, (512, 768), (768, 1), 0); del buf84  # reuse
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg26_1, reinterpret_tensor(buf85, (512, 768), (768, 1), 0), arg27_1, alpha=1, beta=1, out=buf86)
    del arg26_1
    del arg27_1
    buf87 = buf72; del buf72  # reuse
    buf88 = buf71; del buf71  # reuse
    buf90 = reinterpret_tensor(buf85, (2, 256, 768), (196608, 768, 1), 0); del buf85  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf86.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()))
    del arg112_1
    del arg113_1
    buf91 = reinterpret_tensor(buf82, (512, 3072), (3072, 1), 0); del buf82  # reuse
    # Source Nodes: [x_28], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf90, (512, 768), (768, 1), 0), arg29_1, alpha=1, beta=1, out=buf91)
    del arg28_1
    del arg29_1
    buf92 = reinterpret_tensor(buf91, (2, 256, 3072), (786432, 3072, 1), 0); del buf91  # reuse
    cpp_fused_add_mul_pow_tanh_23(c_void_p(buf92.data_ptr()))
    buf93 = reinterpret_tensor(buf90, (512, 768), (768, 1), 0); del buf90  # reuse
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg30_1, reinterpret_tensor(buf92, (512, 3072), (3072, 1), 0), arg31_1, alpha=1, beta=1, out=buf93)
    del arg30_1
    del arg31_1
    buf94 = buf88; del buf88  # reuse
    buf95 = buf87; del buf87  # reuse
    buf97 = reinterpret_tensor(buf46, (2, 256, 768), (196608, 768, 1), 0); del buf46  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf86.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()))
    del arg114_1
    del arg115_1
    buf98 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg32_1, reinterpret_tensor(buf97, (512, 768), (768, 1), 0), arg33_1, alpha=1, beta=1, out=buf98)
    del arg32_1
    del arg33_1
    buf99 = reinterpret_tensor(buf97, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf97  # reuse
    buf100 = reinterpret_tensor(buf39, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf39  # reuse
    cpp_fused_clone_25(c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    buf101 = reinterpret_tensor(buf92, (24, 256, 256), (65536, 256, 1), 0); del buf92  # reuse
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf99, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf100, (24, 64, 256), (16384, 256, 1), 0), out=buf101)
    buf102 = buf81; del buf81  # reuse
    buf103 = reinterpret_tensor(buf101, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf101  # reuse
    buf104 = buf79; del buf79  # reuse
    buf105 = buf103; del buf103  # reuse
    buf106 = buf99; del buf99  # reuse
    cpp_fused__softmax_clone_div_full_where_26(c_void_p(buf105.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    del arg153_1
    buf107 = reinterpret_tensor(buf100, (24, 256, 64), (16384, 64, 1), 0); del buf100  # reuse
    # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf105, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf106, (24, 256, 64), (16384, 64, 1), 0), out=buf107)
    buf108 = reinterpret_tensor(buf106, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf106  # reuse
    cpp_fused_clone_27(c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()))
    buf109 = reinterpret_tensor(buf107, (512, 768), (768, 1), 0); del buf107  # reuse
    # Source Nodes: [x_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf108, (512, 768), (768, 1), 0), arg35_1, alpha=1, beta=1, out=buf109)
    del arg34_1
    del arg35_1
    buf110 = buf95; del buf95  # reuse
    buf111 = buf94; del buf94  # reuse
    buf113 = reinterpret_tensor(buf108, (2, 256, 768), (196608, 768, 1), 0); del buf108  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf109.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()))
    del arg116_1
    del arg117_1
    buf114 = reinterpret_tensor(buf105, (512, 3072), (3072, 1), 0); del buf105  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg36_1, reinterpret_tensor(buf113, (512, 768), (768, 1), 0), arg37_1, alpha=1, beta=1, out=buf114)
    del arg36_1
    del arg37_1
    buf115 = reinterpret_tensor(buf114, (2, 256, 3072), (786432, 3072, 1), 0); del buf114  # reuse
    cpp_fused_add_mul_pow_tanh_29(c_void_p(buf115.data_ptr()))
    buf116 = reinterpret_tensor(buf113, (512, 768), (768, 1), 0); del buf113  # reuse
    # Source Nodes: [x_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf115, (512, 3072), (3072, 1), 0), arg39_1, alpha=1, beta=1, out=buf116)
    del arg38_1
    del arg39_1
    buf117 = reinterpret_tensor(buf116, (2, 256, 768), (196608, 768, 1), 0); del buf116  # reuse
    buf118 = buf111; del buf111  # reuse
    buf119 = buf110; del buf110  # reuse
    buf121 = buf23; del buf23  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf117.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    del arg118_1
    del arg119_1
    buf122 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf121, (512, 768), (768, 1), 0), arg41_1, alpha=1, beta=1, out=buf122)
    del arg40_1
    del arg41_1
    buf123 = reinterpret_tensor(buf121, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf121  # reuse
    buf124 = reinterpret_tensor(buf93, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf93  # reuse
    cpp_fused_clone_31(c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = reinterpret_tensor(buf115, (24, 256, 256), (65536, 256, 1), 0); del buf115  # reuse
    # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf123, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf124, (24, 64, 256), (16384, 256, 1), 0), out=buf125)
    buf126 = buf104; del buf104  # reuse
    buf127 = reinterpret_tensor(buf125, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf125  # reuse
    buf128 = buf102; del buf102  # reuse
    buf129 = buf127; del buf127  # reuse
    buf130 = reinterpret_tensor(buf124, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf124  # reuse
    cpp_fused__softmax_clone_div_full_where_32(c_void_p(buf129.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg154_1
    buf131 = reinterpret_tensor(buf123, (24, 256, 64), (16384, 64, 1), 0); del buf123  # reuse
    # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf129, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf130, (24, 256, 64), (16384, 64, 1), 0), out=buf131)
    buf132 = reinterpret_tensor(buf130, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf130  # reuse
    cpp_fused_clone_33(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf131, (512, 768), (768, 1), 0); del buf131  # reuse
    # Source Nodes: [x_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg42_1, reinterpret_tensor(buf132, (512, 768), (768, 1), 0), arg43_1, alpha=1, beta=1, out=buf133)
    del arg42_1
    del arg43_1
    buf134 = buf119; del buf119  # reuse
    buf135 = buf118; del buf118  # reuse
    buf137 = reinterpret_tensor(buf132, (2, 256, 768), (196608, 768, 1), 0); del buf132  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf133.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf137.data_ptr()))
    del arg120_1
    del arg121_1
    buf138 = reinterpret_tensor(buf129, (512, 3072), (3072, 1), 0); del buf129  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg44_1, reinterpret_tensor(buf137, (512, 768), (768, 1), 0), arg45_1, alpha=1, beta=1, out=buf138)
    del arg44_1
    del arg45_1
    buf139 = reinterpret_tensor(buf138, (2, 256, 3072), (786432, 3072, 1), 0); del buf138  # reuse
    cpp_fused_add_mul_pow_tanh_35(c_void_p(buf139.data_ptr()))
    buf140 = reinterpret_tensor(buf137, (512, 768), (768, 1), 0); del buf137  # reuse
    # Source Nodes: [x_46], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg46_1, reinterpret_tensor(buf139, (512, 3072), (3072, 1), 0), arg47_1, alpha=1, beta=1, out=buf140)
    del arg46_1
    del arg47_1
    buf141 = buf135; del buf135  # reuse
    buf142 = buf134; del buf134  # reuse
    buf144 = reinterpret_tensor(buf86, (2, 256, 768), (196608, 768, 1), 0); del buf86  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf133.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg122_1
    del arg123_1
    buf145 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg48_1, reinterpret_tensor(buf144, (512, 768), (768, 1), 0), arg49_1, alpha=1, beta=1, out=buf145)
    del arg48_1
    del arg49_1
    buf146 = reinterpret_tensor(buf144, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf144  # reuse
    buf147 = reinterpret_tensor(buf70, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf70  # reuse
    cpp_fused_clone_37(c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    buf148 = reinterpret_tensor(buf139, (24, 256, 256), (65536, 256, 1), 0); del buf139  # reuse
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf146, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf147, (24, 64, 256), (16384, 256, 1), 0), out=buf148)
    buf149 = buf128; del buf128  # reuse
    buf150 = reinterpret_tensor(buf148, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf148  # reuse
    buf151 = buf126; del buf126  # reuse
    buf152 = buf150; del buf150  # reuse
    buf153 = reinterpret_tensor(buf147, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf147  # reuse
    cpp_fused__softmax_clone_div_full_where_38(c_void_p(buf152.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg155_1
    buf154 = reinterpret_tensor(buf146, (24, 256, 64), (16384, 64, 1), 0); del buf146  # reuse
    # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf152, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf153, (24, 256, 64), (16384, 64, 1), 0), out=buf154)
    buf155 = reinterpret_tensor(buf153, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf153  # reuse
    cpp_fused_clone_39(c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    buf156 = reinterpret_tensor(buf154, (512, 768), (768, 1), 0); del buf154  # reuse
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf155, (512, 768), (768, 1), 0), arg51_1, alpha=1, beta=1, out=buf156)
    del arg50_1
    del arg51_1
    buf157 = buf142; del buf142  # reuse
    buf158 = buf141; del buf141  # reuse
    buf160 = reinterpret_tensor(buf155, (2, 256, 768), (196608, 768, 1), 0); del buf155  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf156.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()))
    del arg124_1
    del arg125_1
    buf161 = reinterpret_tensor(buf152, (512, 3072), (3072, 1), 0); del buf152  # reuse
    # Source Nodes: [x_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg52_1, reinterpret_tensor(buf160, (512, 768), (768, 1), 0), arg53_1, alpha=1, beta=1, out=buf161)
    del arg52_1
    del arg53_1
    buf162 = reinterpret_tensor(buf161, (2, 256, 3072), (786432, 3072, 1), 0); del buf161  # reuse
    cpp_fused_add_mul_pow_tanh_41(c_void_p(buf162.data_ptr()))
    buf163 = reinterpret_tensor(buf160, (512, 768), (768, 1), 0); del buf160  # reuse
    # Source Nodes: [x_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg54_1, reinterpret_tensor(buf162, (512, 3072), (3072, 1), 0), arg55_1, alpha=1, beta=1, out=buf163)
    del arg54_1
    del arg55_1
    buf164 = reinterpret_tensor(buf163, (2, 256, 768), (196608, 768, 1), 0); del buf163  # reuse
    buf165 = buf158; del buf158  # reuse
    buf166 = buf157; del buf157  # reuse
    buf168 = reinterpret_tensor(buf109, (2, 256, 768), (196608, 768, 1), 0); del buf109  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf164.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()))
    del arg126_1
    del arg127_1
    buf169 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, reinterpret_tensor(buf168, (512, 768), (768, 1), 0), arg57_1, alpha=1, beta=1, out=buf169)
    del arg56_1
    del arg57_1
    buf170 = reinterpret_tensor(buf168, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf168  # reuse
    buf171 = reinterpret_tensor(buf156, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf156  # reuse
    cpp_fused_clone_43(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    buf172 = reinterpret_tensor(buf162, (24, 256, 256), (65536, 256, 1), 0); del buf162  # reuse
    # Source Nodes: [attn_weights_49], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf170, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf171, (24, 64, 256), (16384, 256, 1), 0), out=buf172)
    buf173 = buf151; del buf151  # reuse
    buf174 = reinterpret_tensor(buf172, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf172  # reuse
    buf175 = buf149; del buf149  # reuse
    buf176 = buf174; del buf174  # reuse
    buf177 = reinterpret_tensor(buf171, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf171  # reuse
    cpp_fused__softmax_clone_div_full_where_44(c_void_p(buf176.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg156_1
    buf178 = reinterpret_tensor(buf170, (24, 256, 64), (16384, 64, 1), 0); del buf170  # reuse
    # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf176, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf177, (24, 256, 64), (16384, 64, 1), 0), out=buf178)
    buf179 = reinterpret_tensor(buf177, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf177  # reuse
    cpp_fused_clone_45(c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    buf180 = reinterpret_tensor(buf178, (512, 768), (768, 1), 0); del buf178  # reuse
    # Source Nodes: [x_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf179, (512, 768), (768, 1), 0), arg59_1, alpha=1, beta=1, out=buf180)
    del arg58_1
    del arg59_1
    buf181 = buf166; del buf166  # reuse
    buf182 = buf165; del buf165  # reuse
    buf184 = reinterpret_tensor(buf179, (2, 256, 768), (196608, 768, 1), 0); del buf179  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf180.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg128_1
    del arg129_1
    buf185 = reinterpret_tensor(buf176, (512, 3072), (3072, 1), 0); del buf176  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf184, (512, 768), (768, 1), 0), arg61_1, alpha=1, beta=1, out=buf185)
    del arg60_1
    del arg61_1
    buf186 = reinterpret_tensor(buf185, (2, 256, 3072), (786432, 3072, 1), 0); del buf185  # reuse
    cpp_fused_add_mul_pow_tanh_47(c_void_p(buf186.data_ptr()))
    buf187 = reinterpret_tensor(buf184, (512, 768), (768, 1), 0); del buf184  # reuse
    # Source Nodes: [x_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg62_1, reinterpret_tensor(buf186, (512, 3072), (3072, 1), 0), arg63_1, alpha=1, beta=1, out=buf187)
    del arg62_1
    del arg63_1
    buf188 = buf182; del buf182  # reuse
    buf189 = buf181; del buf181  # reuse
    buf191 = reinterpret_tensor(buf140, (2, 256, 768), (196608, 768, 1), 0); del buf140  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf180.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()))
    del arg130_1
    del arg131_1
    buf192 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf191, (512, 768), (768, 1), 0), arg65_1, alpha=1, beta=1, out=buf192)
    del arg64_1
    del arg65_1
    buf193 = reinterpret_tensor(buf191, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf191  # reuse
    buf194 = reinterpret_tensor(buf133, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf133  # reuse
    cpp_fused_clone_49(c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    buf195 = reinterpret_tensor(buf186, (24, 256, 256), (65536, 256, 1), 0); del buf186  # reuse
    # Source Nodes: [attn_weights_56], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf193, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf194, (24, 64, 256), (16384, 256, 1), 0), out=buf195)
    buf196 = buf175; del buf175  # reuse
    buf197 = reinterpret_tensor(buf195, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf195  # reuse
    buf198 = buf173; del buf173  # reuse
    buf199 = buf197; del buf197  # reuse
    buf200 = reinterpret_tensor(buf194, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf194  # reuse
    cpp_fused__softmax_clone_div_full_where_50(c_void_p(buf199.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf200.data_ptr()))
    del arg157_1
    buf201 = reinterpret_tensor(buf193, (24, 256, 64), (16384, 64, 1), 0); del buf193  # reuse
    # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf199, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf200, (24, 256, 64), (16384, 64, 1), 0), out=buf201)
    buf202 = reinterpret_tensor(buf200, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf200  # reuse
    cpp_fused_clone_51(c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf201, (512, 768), (768, 1), 0); del buf201  # reuse
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf202, (512, 768), (768, 1), 0), arg67_1, alpha=1, beta=1, out=buf203)
    del arg66_1
    del arg67_1
    buf204 = buf189; del buf189  # reuse
    buf205 = buf188; del buf188  # reuse
    buf207 = reinterpret_tensor(buf202, (2, 256, 768), (196608, 768, 1), 0); del buf202  # reuse
    cpp_fused_add_native_layer_norm_52(c_void_p(buf203.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()))
    del arg132_1
    del arg133_1
    buf208 = reinterpret_tensor(buf199, (512, 3072), (3072, 1), 0); del buf199  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg68_1, reinterpret_tensor(buf207, (512, 768), (768, 1), 0), arg69_1, alpha=1, beta=1, out=buf208)
    del arg68_1
    del arg69_1
    buf209 = reinterpret_tensor(buf208, (2, 256, 3072), (786432, 3072, 1), 0); del buf208  # reuse
    cpp_fused_add_mul_pow_tanh_53(c_void_p(buf209.data_ptr()))
    buf210 = reinterpret_tensor(buf207, (512, 768), (768, 1), 0); del buf207  # reuse
    # Source Nodes: [x_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf209, (512, 3072), (3072, 1), 0), arg71_1, alpha=1, beta=1, out=buf210)
    del arg70_1
    del arg71_1
    buf211 = reinterpret_tensor(buf210, (2, 256, 768), (196608, 768, 1), 0); del buf210  # reuse
    buf212 = buf205; del buf205  # reuse
    buf213 = buf204; del buf204  # reuse
    buf215 = buf117; del buf117  # reuse
    cpp_fused_add_native_layer_norm_54(c_void_p(buf211.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()))
    del arg134_1
    del arg135_1
    buf216 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg72_1, reinterpret_tensor(buf215, (512, 768), (768, 1), 0), arg73_1, alpha=1, beta=1, out=buf216)
    del arg72_1
    del arg73_1
    buf217 = reinterpret_tensor(buf215, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf215  # reuse
    buf218 = reinterpret_tensor(buf203, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf203  # reuse
    cpp_fused_clone_55(c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    buf219 = reinterpret_tensor(buf209, (24, 256, 256), (65536, 256, 1), 0); del buf209  # reuse
    # Source Nodes: [attn_weights_63], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf217, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf218, (24, 64, 256), (16384, 256, 1), 0), out=buf219)
    buf220 = buf198; del buf198  # reuse
    buf221 = reinterpret_tensor(buf219, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf219  # reuse
    buf222 = buf196; del buf196  # reuse
    buf223 = buf221; del buf221  # reuse
    buf224 = reinterpret_tensor(buf218, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf218  # reuse
    cpp_fused__softmax_clone_div_full_where_56(c_void_p(buf223.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()))
    del arg158_1
    buf225 = reinterpret_tensor(buf217, (24, 256, 64), (16384, 64, 1), 0); del buf217  # reuse
    # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf223, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf224, (24, 256, 64), (16384, 64, 1), 0), out=buf225)
    buf226 = reinterpret_tensor(buf224, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf224  # reuse
    cpp_fused_clone_57(c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    buf227 = reinterpret_tensor(buf225, (512, 768), (768, 1), 0); del buf225  # reuse
    # Source Nodes: [x_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg74_1, reinterpret_tensor(buf226, (512, 768), (768, 1), 0), arg75_1, alpha=1, beta=1, out=buf227)
    del arg74_1
    del arg75_1
    buf228 = buf213; del buf213  # reuse
    buf229 = buf212; del buf212  # reuse
    buf231 = reinterpret_tensor(buf226, (2, 256, 768), (196608, 768, 1), 0); del buf226  # reuse
    cpp_fused_add_native_layer_norm_58(c_void_p(buf227.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()))
    del arg136_1
    del arg137_1
    buf232 = reinterpret_tensor(buf223, (512, 3072), (3072, 1), 0); del buf223  # reuse
    # Source Nodes: [x_76], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg76_1, reinterpret_tensor(buf231, (512, 768), (768, 1), 0), arg77_1, alpha=1, beta=1, out=buf232)
    del arg76_1
    del arg77_1
    buf233 = reinterpret_tensor(buf232, (2, 256, 3072), (786432, 3072, 1), 0); del buf232  # reuse
    cpp_fused_add_mul_pow_tanh_59(c_void_p(buf233.data_ptr()))
    buf234 = reinterpret_tensor(buf231, (512, 768), (768, 1), 0); del buf231  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf233, (512, 3072), (3072, 1), 0), arg79_1, alpha=1, beta=1, out=buf234)
    del arg78_1
    del arg79_1
    buf235 = buf229; del buf229  # reuse
    buf236 = buf228; del buf228  # reuse
    buf238 = reinterpret_tensor(buf187, (2, 256, 768), (196608, 768, 1), 0); del buf187  # reuse
    cpp_fused_add_native_layer_norm_60(c_void_p(buf227.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()))
    del arg138_1
    del arg139_1
    buf239 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf238, (512, 768), (768, 1), 0), arg81_1, alpha=1, beta=1, out=buf239)
    del arg80_1
    del arg81_1
    buf240 = reinterpret_tensor(buf238, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf238  # reuse
    buf241 = reinterpret_tensor(buf180, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf180  # reuse
    cpp_fused_clone_61(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    buf242 = reinterpret_tensor(buf233, (24, 256, 256), (65536, 256, 1), 0); del buf233  # reuse
    # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf240, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf241, (24, 64, 256), (16384, 256, 1), 0), out=buf242)
    buf243 = buf222; del buf222  # reuse
    buf244 = reinterpret_tensor(buf242, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf242  # reuse
    buf245 = buf220; del buf220  # reuse
    buf246 = buf244; del buf244  # reuse
    buf247 = reinterpret_tensor(buf241, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf241  # reuse
    cpp_fused__softmax_clone_div_full_where_62(c_void_p(buf246.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg159_1
    buf248 = reinterpret_tensor(buf240, (24, 256, 64), (16384, 64, 1), 0); del buf240  # reuse
    # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf246, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf247, (24, 256, 64), (16384, 64, 1), 0), out=buf248)
    buf249 = reinterpret_tensor(buf247, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf247  # reuse
    cpp_fused_clone_63(c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    buf250 = reinterpret_tensor(buf248, (512, 768), (768, 1), 0); del buf248  # reuse
    # Source Nodes: [x_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf249, (512, 768), (768, 1), 0), arg83_1, alpha=1, beta=1, out=buf250)
    del arg82_1
    del arg83_1
    buf251 = buf236; del buf236  # reuse
    buf252 = buf235; del buf235  # reuse
    buf254 = reinterpret_tensor(buf249, (2, 256, 768), (196608, 768, 1), 0); del buf249  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf250.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf254.data_ptr()))
    del arg140_1
    del arg141_1
    buf255 = reinterpret_tensor(buf246, (512, 3072), (3072, 1), 0); del buf246  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg84_1, reinterpret_tensor(buf254, (512, 768), (768, 1), 0), arg85_1, alpha=1, beta=1, out=buf255)
    del arg84_1
    del arg85_1
    buf256 = reinterpret_tensor(buf255, (2, 256, 3072), (786432, 3072, 1), 0); del buf255  # reuse
    cpp_fused_add_mul_pow_tanh_65(c_void_p(buf256.data_ptr()))
    buf257 = reinterpret_tensor(buf254, (512, 768), (768, 1), 0); del buf254  # reuse
    # Source Nodes: [x_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf256, (512, 3072), (3072, 1), 0), arg87_1, alpha=1, beta=1, out=buf257)
    del arg86_1
    del arg87_1
    buf258 = reinterpret_tensor(buf257, (2, 256, 768), (196608, 768, 1), 0); del buf257  # reuse
    buf259 = buf252; del buf252  # reuse
    buf260 = buf251; del buf251  # reuse
    buf262 = buf164; del buf164  # reuse
    cpp_fused_add_native_layer_norm_66(c_void_p(buf258.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()))
    del arg142_1
    del arg143_1
    del buf211
    del buf227
    buf263 = empty((512, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf262, (512, 768), (768, 1), 0), arg89_1, alpha=1, beta=1, out=buf263)
    del arg88_1
    del arg89_1
    buf264 = reinterpret_tensor(buf262, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf262  # reuse
    buf265 = reinterpret_tensor(buf250, (2, 12, 64, 256), (196608, 16384, 256, 1), 0); del buf250  # reuse
    cpp_fused_clone_67(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = reinterpret_tensor(buf256, (24, 256, 256), (65536, 256, 1), 0); del buf256  # reuse
    # Source Nodes: [attn_weights_77], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf264, (24, 256, 64), (16384, 64, 1), 0), reinterpret_tensor(buf265, (24, 64, 256), (16384, 256, 1), 0), out=buf266)
    buf267 = buf245; del buf245  # reuse
    buf268 = reinterpret_tensor(buf266, (2, 12, 256, 256), (786432, 65536, 256, 1), 0); del buf266  # reuse
    buf269 = buf243; del buf243  # reuse
    buf270 = buf268; del buf268  # reuse
    buf271 = reinterpret_tensor(buf265, (2, 12, 256, 64), (196608, 16384, 64, 1), 0); del buf265  # reuse
    cpp_fused__softmax_clone_div_full_where_68(c_void_p(buf270.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()))
    del arg160_1
    del buf267
    del buf269
    buf272 = reinterpret_tensor(buf264, (24, 256, 64), (16384, 64, 1), 0); del buf264  # reuse
    # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf270, (24, 256, 256), (65536, 256, 1), 0), reinterpret_tensor(buf271, (24, 256, 64), (16384, 64, 1), 0), out=buf272)
    buf273 = reinterpret_tensor(buf271, (2, 256, 12, 64), (196608, 768, 64, 1), 0); del buf271  # reuse
    cpp_fused_clone_69(c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    buf274 = reinterpret_tensor(buf272, (512, 768), (768, 1), 0); del buf272  # reuse
    # Source Nodes: [x_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf273, (512, 768), (768, 1), 0), arg91_1, alpha=1, beta=1, out=buf274)
    del arg90_1
    del arg91_1
    buf275 = buf260; del buf260  # reuse
    buf276 = buf259; del buf259  # reuse
    buf278 = reinterpret_tensor(buf273, (2, 256, 768), (196608, 768, 1), 0); del buf273  # reuse
    cpp_fused_add_native_layer_norm_70(c_void_p(buf274.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf278.data_ptr()))
    del arg144_1
    del arg145_1
    buf279 = reinterpret_tensor(buf270, (512, 3072), (3072, 1), 0); del buf270  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf278, (512, 768), (768, 1), 0), arg93_1, alpha=1, beta=1, out=buf279)
    del arg92_1
    del arg93_1
    buf280 = reinterpret_tensor(buf279, (2, 256, 3072), (786432, 3072, 1), 0); del buf279  # reuse
    cpp_fused_add_mul_pow_tanh_71(c_void_p(buf280.data_ptr()))
    buf281 = reinterpret_tensor(buf278, (512, 768), (768, 1), 0); del buf278  # reuse
    # Source Nodes: [x_94], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg94_1, reinterpret_tensor(buf280, (512, 3072), (3072, 1), 0), arg95_1, alpha=1, beta=1, out=buf281)
    del arg94_1
    del arg95_1
    del buf280
    buf282 = buf276; del buf276  # reuse
    buf283 = buf275; del buf275  # reuse
    buf285 = reinterpret_tensor(buf234, (2, 256, 768), (196608, 768, 1), 0); del buf234  # reuse
    cpp_fused_add_native_layer_norm_72(c_void_p(buf274.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()))
    del arg146_1
    del arg147_1
    del buf258
    del buf274
    del buf281
    del buf282
    del buf283
    buf286 = empty((512, 50257), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf285, (512, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 50257), (1, 768), 0), out=buf286)
    del arg148_1
    return (reinterpret_tensor(buf286, (2, 256, 50257), (12865792, 50257, 1), 0), reinterpret_tensor(buf4, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf4, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf28, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf28, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf51, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf51, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf75, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf75, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf98, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf98, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf122, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf122, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf145, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf145, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf169, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf169, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf192, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf192, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf216, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf216, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf239, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf239, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), reinterpret_tensor(buf263, (2, 12, 256, 64), (589824, 64, 2304, 1), 768), reinterpret_tensor(buf263, (2, 12, 256, 64), (589824, 64, 2304, 1), 1536), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, 2304), (2304, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((50257, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1024, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((50257, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg150_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg151_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg152_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg153_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg154_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg155_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg156_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg157_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg158_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg159_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg160_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cpu', dtype=torch.bool)
    arg161_1 = rand_strided((2, 256), (256, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('hf_GPT2', benchmark_compiled_module)
