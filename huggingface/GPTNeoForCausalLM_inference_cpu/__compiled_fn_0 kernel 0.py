
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 50257);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50257L), "index out of bounds: 0 <= tmp3 < 50257L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*tmp3)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 50257);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 50257L), "index out of bounds: 0 <= tmp3 < 50257L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*tmp3)));
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                        auto tmp3 = tmp1 < 0;
                        auto tmp4 = tmp3 ? tmp2 : tmp1;
                        TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*tmp4)));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp0 + tmp7;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = in_ptr1[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                    auto tmp3 = tmp1 < 0;
                    auto tmp4 = tmp3 ? tmp2 : tmp1;
                    TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*tmp4)));
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp8 = tmp0 + tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp13 = static_cast<float>(2048.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = static_cast<float>(1e-05);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = 1 / std::sqrt(tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp21 = tmp19 * tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = in_ptr1[static_cast<long>(x0)];
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = decltype(tmp1)(tmp1 + 50257);
                        auto tmp3 = tmp1 < 0;
                        auto tmp4 = tmp3 ? tmp2 : tmp1;
                        TORCH_CHECK((0 <= tmp4) & (tmp4 < 50257L), "index out of bounds: 0 <= tmp4 < 50257L")
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*tmp4)));
                        auto tmp7 = tmp5 + tmp6;
                        auto tmp8 = tmp0 + tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp10);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_75 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_81 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_87 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_90 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_93 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_94 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_96 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_99 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_103 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_105 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_106 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_108 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_111 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_112 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_114 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_117 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_118 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_120 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_123 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_124 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_126 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_127 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_129 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_130 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_132 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_133 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_135 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_136 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(2048.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_138 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(262144L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_lift_fresh_where_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
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
                            auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp4);
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
                        auto tmp0 = flag_to_float_vec(in_ptr0 + static_cast<long>(x2 + (2048L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp8 = tmp7.exp();
                        tmp8.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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
    }
}
''')


cpp_fused__softmax_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_141 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0) + (16384L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (2048L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_142 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(2048.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_143 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_144 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(2048.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_nll_loss_forward_145 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(127L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50257L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(50256L); x1<static_cast<long>(50257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50257L*x0))];
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(127L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50257L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(50256L); x1<static_cast<long>(50257L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (50257L*x0))];
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp3 = std::exp(tmp2);
                        tmp_acc0 = tmp_acc0 + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    float tmp_acc0 = 0;
                    long tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(127L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp9 = out_ptr0[static_cast<long>(x0)];
                        auto tmp11 = out_ptr1[static_cast<long>(x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = static_cast<long>(0);
                        auto tmp4 = tmp2 ? tmp0 : tmp3;
                        auto tmp5 = decltype(tmp4)(tmp4 + 50257);
                        auto tmp6 = tmp4 < 0;
                        auto tmp7 = tmp6 ? tmp5 : tmp4;
                        TORCH_CHECK((0 <= tmp7) & (tmp7 < 50257L), "index out of bounds: 0 <= tmp7 < 50257L")
                        auto tmp8 = in_ptr0[static_cast<long>(tmp7 + (50257L*x0))];
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp12 = std::log(tmp11);
                        auto tmp13 = decltype(tmp10)(tmp10 - tmp12);
                        auto tmp14 = decltype(tmp13)(-tmp13);
                        auto tmp15 = static_cast<float>(0.0);
                        auto tmp16 = tmp2 ? tmp14 : tmp15;
                        auto tmp17 = c10::convert<long>(tmp2);
                        tmp_acc0 = tmp_acc0 + tmp16;
                        tmp_acc1 = tmp_acc1 + tmp17;
                    }
                    out_ptr2[static_cast<long>(0L)] = tmp_acc0;
                    out_ptr3[static_cast<long>(0L)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = out_ptr2[static_cast<long>(0L)];
                auto tmp1 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                in_out_ptr0[static_cast<long>(0L)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1 = args
    args.clear()
    assert_size_stride(arg0_1, (50257, 2048), (2048, 1))
    assert_size_stride(arg1_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg2_1, (2048, ), (1, ))
    assert_size_stride(arg3_1, (2048, ), (1, ))
    assert_size_stride(arg4_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg5_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg6_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg7_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg8_1, (2048, ), (1, ))
    assert_size_stride(arg9_1, (2048, ), (1, ))
    assert_size_stride(arg10_1, (2048, ), (1, ))
    assert_size_stride(arg11_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg12_1, (8192, ), (1, ))
    assert_size_stride(arg13_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg14_1, (2048, ), (1, ))
    assert_size_stride(arg15_1, (2048, ), (1, ))
    assert_size_stride(arg16_1, (2048, ), (1, ))
    assert_size_stride(arg17_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg18_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg19_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg20_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg21_1, (2048, ), (1, ))
    assert_size_stride(arg22_1, (2048, ), (1, ))
    assert_size_stride(arg23_1, (2048, ), (1, ))
    assert_size_stride(arg24_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg25_1, (8192, ), (1, ))
    assert_size_stride(arg26_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg27_1, (2048, ), (1, ))
    assert_size_stride(arg28_1, (2048, ), (1, ))
    assert_size_stride(arg29_1, (2048, ), (1, ))
    assert_size_stride(arg30_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg31_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg32_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg33_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg34_1, (2048, ), (1, ))
    assert_size_stride(arg35_1, (2048, ), (1, ))
    assert_size_stride(arg36_1, (2048, ), (1, ))
    assert_size_stride(arg37_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg38_1, (8192, ), (1, ))
    assert_size_stride(arg39_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg40_1, (2048, ), (1, ))
    assert_size_stride(arg41_1, (2048, ), (1, ))
    assert_size_stride(arg42_1, (2048, ), (1, ))
    assert_size_stride(arg43_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg44_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg45_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg46_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg47_1, (2048, ), (1, ))
    assert_size_stride(arg48_1, (2048, ), (1, ))
    assert_size_stride(arg49_1, (2048, ), (1, ))
    assert_size_stride(arg50_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg51_1, (8192, ), (1, ))
    assert_size_stride(arg52_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg53_1, (2048, ), (1, ))
    assert_size_stride(arg54_1, (2048, ), (1, ))
    assert_size_stride(arg55_1, (2048, ), (1, ))
    assert_size_stride(arg56_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg57_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg58_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg59_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg60_1, (2048, ), (1, ))
    assert_size_stride(arg61_1, (2048, ), (1, ))
    assert_size_stride(arg62_1, (2048, ), (1, ))
    assert_size_stride(arg63_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg64_1, (8192, ), (1, ))
    assert_size_stride(arg65_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg66_1, (2048, ), (1, ))
    assert_size_stride(arg67_1, (2048, ), (1, ))
    assert_size_stride(arg68_1, (2048, ), (1, ))
    assert_size_stride(arg69_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg70_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg71_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg72_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg73_1, (2048, ), (1, ))
    assert_size_stride(arg74_1, (2048, ), (1, ))
    assert_size_stride(arg75_1, (2048, ), (1, ))
    assert_size_stride(arg76_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg77_1, (8192, ), (1, ))
    assert_size_stride(arg78_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg79_1, (2048, ), (1, ))
    assert_size_stride(arg80_1, (2048, ), (1, ))
    assert_size_stride(arg81_1, (2048, ), (1, ))
    assert_size_stride(arg82_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg83_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg84_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg85_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg86_1, (2048, ), (1, ))
    assert_size_stride(arg87_1, (2048, ), (1, ))
    assert_size_stride(arg88_1, (2048, ), (1, ))
    assert_size_stride(arg89_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg90_1, (8192, ), (1, ))
    assert_size_stride(arg91_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg92_1, (2048, ), (1, ))
    assert_size_stride(arg93_1, (2048, ), (1, ))
    assert_size_stride(arg94_1, (2048, ), (1, ))
    assert_size_stride(arg95_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg96_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg97_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg98_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg99_1, (2048, ), (1, ))
    assert_size_stride(arg100_1, (2048, ), (1, ))
    assert_size_stride(arg101_1, (2048, ), (1, ))
    assert_size_stride(arg102_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg103_1, (8192, ), (1, ))
    assert_size_stride(arg104_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg105_1, (2048, ), (1, ))
    assert_size_stride(arg106_1, (2048, ), (1, ))
    assert_size_stride(arg107_1, (2048, ), (1, ))
    assert_size_stride(arg108_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg109_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg110_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg111_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg112_1, (2048, ), (1, ))
    assert_size_stride(arg113_1, (2048, ), (1, ))
    assert_size_stride(arg114_1, (2048, ), (1, ))
    assert_size_stride(arg115_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg116_1, (8192, ), (1, ))
    assert_size_stride(arg117_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg118_1, (2048, ), (1, ))
    assert_size_stride(arg119_1, (2048, ), (1, ))
    assert_size_stride(arg120_1, (2048, ), (1, ))
    assert_size_stride(arg121_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg122_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg123_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg124_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg125_1, (2048, ), (1, ))
    assert_size_stride(arg126_1, (2048, ), (1, ))
    assert_size_stride(arg127_1, (2048, ), (1, ))
    assert_size_stride(arg128_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg129_1, (8192, ), (1, ))
    assert_size_stride(arg130_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg131_1, (2048, ), (1, ))
    assert_size_stride(arg132_1, (2048, ), (1, ))
    assert_size_stride(arg133_1, (2048, ), (1, ))
    assert_size_stride(arg134_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg135_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg136_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg137_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg138_1, (2048, ), (1, ))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (2048, ), (1, ))
    assert_size_stride(arg141_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg142_1, (8192, ), (1, ))
    assert_size_stride(arg143_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg144_1, (2048, ), (1, ))
    assert_size_stride(arg145_1, (2048, ), (1, ))
    assert_size_stride(arg146_1, (2048, ), (1, ))
    assert_size_stride(arg147_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg148_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg149_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg150_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg151_1, (2048, ), (1, ))
    assert_size_stride(arg152_1, (2048, ), (1, ))
    assert_size_stride(arg153_1, (2048, ), (1, ))
    assert_size_stride(arg154_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg155_1, (8192, ), (1, ))
    assert_size_stride(arg156_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (2048, ), (1, ))
    assert_size_stride(arg160_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg161_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg162_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg163_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (2048, ), (1, ))
    assert_size_stride(arg166_1, (2048, ), (1, ))
    assert_size_stride(arg167_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg168_1, (8192, ), (1, ))
    assert_size_stride(arg169_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg170_1, (2048, ), (1, ))
    assert_size_stride(arg171_1, (2048, ), (1, ))
    assert_size_stride(arg172_1, (2048, ), (1, ))
    assert_size_stride(arg173_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg174_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg175_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg176_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg177_1, (2048, ), (1, ))
    assert_size_stride(arg178_1, (2048, ), (1, ))
    assert_size_stride(arg179_1, (2048, ), (1, ))
    assert_size_stride(arg180_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg181_1, (8192, ), (1, ))
    assert_size_stride(arg182_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg183_1, (2048, ), (1, ))
    assert_size_stride(arg184_1, (2048, ), (1, ))
    assert_size_stride(arg185_1, (2048, ), (1, ))
    assert_size_stride(arg186_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg187_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg188_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg189_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg190_1, (2048, ), (1, ))
    assert_size_stride(arg191_1, (2048, ), (1, ))
    assert_size_stride(arg192_1, (2048, ), (1, ))
    assert_size_stride(arg193_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg194_1, (8192, ), (1, ))
    assert_size_stride(arg195_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (2048, ), (1, ))
    assert_size_stride(arg198_1, (2048, ), (1, ))
    assert_size_stride(arg199_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg200_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg201_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg202_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg203_1, (2048, ), (1, ))
    assert_size_stride(arg204_1, (2048, ), (1, ))
    assert_size_stride(arg205_1, (2048, ), (1, ))
    assert_size_stride(arg206_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg207_1, (8192, ), (1, ))
    assert_size_stride(arg208_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg209_1, (2048, ), (1, ))
    assert_size_stride(arg210_1, (2048, ), (1, ))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg213_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg214_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg215_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg216_1, (2048, ), (1, ))
    assert_size_stride(arg217_1, (2048, ), (1, ))
    assert_size_stride(arg218_1, (2048, ), (1, ))
    assert_size_stride(arg219_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg220_1, (8192, ), (1, ))
    assert_size_stride(arg221_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg222_1, (2048, ), (1, ))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (2048, ), (1, ))
    assert_size_stride(arg225_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg226_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg227_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg228_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (2048, ), (1, ))
    assert_size_stride(arg231_1, (2048, ), (1, ))
    assert_size_stride(arg232_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg233_1, (8192, ), (1, ))
    assert_size_stride(arg234_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg235_1, (2048, ), (1, ))
    assert_size_stride(arg236_1, (2048, ), (1, ))
    assert_size_stride(arg237_1, (2048, ), (1, ))
    assert_size_stride(arg238_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg239_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg240_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg241_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg242_1, (2048, ), (1, ))
    assert_size_stride(arg243_1, (2048, ), (1, ))
    assert_size_stride(arg244_1, (2048, ), (1, ))
    assert_size_stride(arg245_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg246_1, (8192, ), (1, ))
    assert_size_stride(arg247_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg248_1, (2048, ), (1, ))
    assert_size_stride(arg249_1, (2048, ), (1, ))
    assert_size_stride(arg250_1, (2048, ), (1, ))
    assert_size_stride(arg251_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg252_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg253_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg254_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg255_1, (2048, ), (1, ))
    assert_size_stride(arg256_1, (2048, ), (1, ))
    assert_size_stride(arg257_1, (2048, ), (1, ))
    assert_size_stride(arg258_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg259_1, (8192, ), (1, ))
    assert_size_stride(arg260_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg261_1, (2048, ), (1, ))
    assert_size_stride(arg262_1, (2048, ), (1, ))
    assert_size_stride(arg263_1, (2048, ), (1, ))
    assert_size_stride(arg264_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg265_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg266_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg267_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (2048, ), (1, ))
    assert_size_stride(arg270_1, (2048, ), (1, ))
    assert_size_stride(arg271_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg272_1, (8192, ), (1, ))
    assert_size_stride(arg273_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg274_1, (2048, ), (1, ))
    assert_size_stride(arg275_1, (2048, ), (1, ))
    assert_size_stride(arg276_1, (2048, ), (1, ))
    assert_size_stride(arg277_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg278_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg279_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg280_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg281_1, (2048, ), (1, ))
    assert_size_stride(arg282_1, (2048, ), (1, ))
    assert_size_stride(arg283_1, (2048, ), (1, ))
    assert_size_stride(arg284_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg285_1, (8192, ), (1, ))
    assert_size_stride(arg286_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg287_1, (2048, ), (1, ))
    assert_size_stride(arg288_1, (2048, ), (1, ))
    assert_size_stride(arg289_1, (2048, ), (1, ))
    assert_size_stride(arg290_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg291_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg292_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg293_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg294_1, (2048, ), (1, ))
    assert_size_stride(arg295_1, (2048, ), (1, ))
    assert_size_stride(arg296_1, (2048, ), (1, ))
    assert_size_stride(arg297_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg298_1, (8192, ), (1, ))
    assert_size_stride(arg299_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg300_1, (2048, ), (1, ))
    assert_size_stride(arg301_1, (2048, ), (1, ))
    assert_size_stride(arg302_1, (2048, ), (1, ))
    assert_size_stride(arg303_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg304_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg305_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg306_1, (2048, 2048), (2048, 1))
    assert_size_stride(arg307_1, (2048, ), (1, ))
    assert_size_stride(arg308_1, (2048, ), (1, ))
    assert_size_stride(arg309_1, (2048, ), (1, ))
    assert_size_stride(arg310_1, (8192, 2048), (2048, 1))
    assert_size_stride(arg311_1, (8192, ), (1, ))
    assert_size_stride(arg312_1, (2048, 8192), (8192, 1))
    assert_size_stride(arg313_1, (2048, ), (1, ))
    assert_size_stride(arg314_1, (2048, ), (1, ))
    assert_size_stride(arg315_1, (2048, ), (1, ))
    assert_size_stride(arg316_1, (50257, 2048), (2048, 1))
    assert_size_stride(arg317_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg318_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg319_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg320_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg321_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg322_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg323_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg324_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg325_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg326_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg327_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg328_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg329_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg330_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg331_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg332_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg333_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg334_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg335_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg336_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg337_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg338_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg339_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg340_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg341_1, (1, 128), (128, 1))
    assert_size_stride(arg342_1, (1, 128), (128, 1))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_0(c_void_p(arg341_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg2_1
    del arg3_1
    buf4 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg4_1, (2048, 2048), (1, 2048), 0), out=buf4)
    del arg4_1
    buf5 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg5_1, (2048, 2048), (1, 2048), 0), out=buf5)
    del arg5_1
    buf6 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf4, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf5, (16, 128, 128), (128, 1, 2048), 0), out=buf6)
    buf7 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf8 = reinterpret_tensor(buf6, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf6  # reuse
    buf9 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_lift_fresh_where_1(c_void_p(buf8.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()))
    del arg317_1
    buf10 = buf4; del buf4  # reuse
    # Source Nodes: [value], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg6_1, (2048, 2048), (1, 2048), 0), out=buf10)
    del arg6_1
    buf11 = buf8; del buf8  # reuse
    cpp_fused__softmax_2(c_void_p(buf11.data_ptr()), c_void_p(buf9.data_ptr()))
    buf12 = reinterpret_tensor(buf3, (16, 128, 128), (16384, 128, 1), 0); del buf3  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf11, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf10, (16, 128, 128), (128, 2048, 1), 0), out=buf12)
    buf13 = reinterpret_tensor(buf11, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf11  # reuse
    cpp_fused_clone_3(c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    buf14 = reinterpret_tensor(buf12, (128, 2048), (2048, 1), 0); del buf12  # reuse
    # Source Nodes: [attn_output_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf13, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg7_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf14)
    del arg7_1
    del arg8_1
    buf15 = buf1; del buf1  # reuse
    buf16 = buf0; del buf0  # reuse
    buf18 = reinterpret_tensor(buf13, (1, 128, 2048), (262144, 2048, 1), 0); del buf13  # reuse
    cpp_fused_add_embedding_native_layer_norm_4(c_void_p(buf14.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg10_1
    del arg9_1
    buf19 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg12_1, reinterpret_tensor(buf18, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg11_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf19)
    del arg11_1
    del arg12_1
    buf20 = reinterpret_tensor(buf19, (1, 128, 8192), (1048576, 8192, 1), 0); del buf19  # reuse
    cpp_fused_add_mul_pow_tanh_5(c_void_p(buf20.data_ptr()))
    buf21 = reinterpret_tensor(buf18, (128, 2048), (2048, 1), 0); del buf18  # reuse
    # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg14_1, reinterpret_tensor(buf20, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg13_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf21)
    del arg13_1
    del arg14_1
    buf22 = reinterpret_tensor(buf21, (1, 128, 2048), (262144, 2048, 1), 0); del buf21  # reuse
    buf23 = buf16; del buf16  # reuse
    buf24 = buf15; del buf15  # reuse
    buf26 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_6(c_void_p(buf22.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()))
    del arg0_1
    del arg15_1
    del arg16_1
    del arg1_1
    del arg341_1
    buf27 = buf14; del buf14  # reuse
    # Source Nodes: [query_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg17_1, (2048, 2048), (1, 2048), 0), out=buf27)
    del arg17_1
    buf28 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg18_1, (2048, 2048), (1, 2048), 0), out=buf28)
    del arg18_1
    buf29 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf27, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf28, (16, 128, 128), (128, 1, 2048), 0), out=buf29)
    buf30 = buf9; del buf9  # reuse
    buf31 = reinterpret_tensor(buf29, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf29  # reuse
    buf32 = buf7; del buf7  # reuse
    cpp_fused__softmax_lift_fresh_where_7(c_void_p(buf31.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()))
    del arg318_1
    buf33 = buf27; del buf27  # reuse
    # Source Nodes: [value_2], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf26, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg19_1, (2048, 2048), (1, 2048), 0), out=buf33)
    del arg19_1
    buf34 = buf31; del buf31  # reuse
    cpp_fused__softmax_8(c_void_p(buf34.data_ptr()), c_void_p(buf32.data_ptr()))
    buf35 = reinterpret_tensor(buf26, (16, 128, 128), (16384, 128, 1), 0); del buf26  # reuse
    # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf34, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf33, (16, 128, 128), (128, 2048, 1), 0), out=buf35)
    buf36 = reinterpret_tensor(buf34, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf34  # reuse
    cpp_fused_clone_9(c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()))
    buf37 = reinterpret_tensor(buf35, (128, 2048), (2048, 1), 0); del buf35  # reuse
    # Source Nodes: [attn_output_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg21_1, reinterpret_tensor(buf36, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg20_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf37)
    del arg20_1
    del arg21_1
    buf38 = buf24; del buf24  # reuse
    buf39 = buf23; del buf23  # reuse
    buf41 = reinterpret_tensor(buf36, (1, 128, 2048), (262144, 2048, 1), 0); del buf36  # reuse
    cpp_fused_add_native_layer_norm_10(c_void_p(buf37.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()))
    del arg22_1
    del arg23_1
    buf42 = reinterpret_tensor(buf20, (128, 8192), (8192, 1), 0); del buf20  # reuse
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf41, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg24_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf42)
    del arg24_1
    del arg25_1
    buf43 = reinterpret_tensor(buf42, (1, 128, 8192), (1048576, 8192, 1), 0); del buf42  # reuse
    cpp_fused_add_mul_pow_tanh_11(c_void_p(buf43.data_ptr()))
    buf44 = reinterpret_tensor(buf41, (128, 2048), (2048, 1), 0); del buf41  # reuse
    # Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg27_1, reinterpret_tensor(buf43, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg26_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf44)
    del arg26_1
    del arg27_1
    buf45 = buf39; del buf39  # reuse
    buf46 = buf38; del buf38  # reuse
    buf48 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_12(c_void_p(buf37.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg28_1
    del arg29_1
    buf49 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg30_1, (2048, 2048), (1, 2048), 0), out=buf49)
    del arg30_1
    buf50 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg31_1, (2048, 2048), (1, 2048), 0), out=buf50)
    del arg31_1
    buf51 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf50, (16, 128, 128), (128, 1, 2048), 0), out=buf51)
    buf52 = buf32; del buf32  # reuse
    buf53 = reinterpret_tensor(buf51, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf51  # reuse
    buf54 = buf30; del buf30  # reuse
    cpp_fused__softmax_lift_fresh_where_13(c_void_p(buf53.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()))
    del arg319_1
    buf55 = buf49; del buf49  # reuse
    # Source Nodes: [value_4], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf48, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg32_1, (2048, 2048), (1, 2048), 0), out=buf55)
    del arg32_1
    buf56 = buf53; del buf53  # reuse
    cpp_fused__softmax_14(c_void_p(buf56.data_ptr()), c_void_p(buf54.data_ptr()))
    buf57 = reinterpret_tensor(buf48, (16, 128, 128), (16384, 128, 1), 0); del buf48  # reuse
    # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf56, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf55, (16, 128, 128), (128, 2048, 1), 0), out=buf57)
    buf58 = reinterpret_tensor(buf56, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf56  # reuse
    cpp_fused_clone_15(c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = reinterpret_tensor(buf57, (128, 2048), (2048, 1), 0); del buf57  # reuse
    # Source Nodes: [attn_output_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg34_1, reinterpret_tensor(buf58, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg33_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf59)
    del arg33_1
    del arg34_1
    buf60 = buf46; del buf46  # reuse
    buf61 = buf45; del buf45  # reuse
    buf63 = reinterpret_tensor(buf58, (1, 128, 2048), (262144, 2048, 1), 0); del buf58  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf59.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()))
    del arg35_1
    del arg36_1
    buf64 = reinterpret_tensor(buf43, (128, 8192), (8192, 1), 0); del buf43  # reuse
    # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf63, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg37_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf64)
    del arg37_1
    del arg38_1
    buf65 = reinterpret_tensor(buf64, (1, 128, 8192), (1048576, 8192, 1), 0); del buf64  # reuse
    cpp_fused_add_mul_pow_tanh_17(c_void_p(buf65.data_ptr()))
    buf66 = reinterpret_tensor(buf63, (128, 2048), (2048, 1), 0); del buf63  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf65, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg39_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf66)
    del arg39_1
    del arg40_1
    buf67 = reinterpret_tensor(buf66, (1, 128, 2048), (262144, 2048, 1), 0); del buf66  # reuse
    buf68 = buf61; del buf61  # reuse
    buf69 = buf60; del buf60  # reuse
    buf71 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_18(c_void_p(buf67.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()))
    del arg41_1
    del arg42_1
    buf72 = buf59; del buf59  # reuse
    # Source Nodes: [query_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg43_1, (2048, 2048), (1, 2048), 0), out=buf72)
    del arg43_1
    buf73 = buf44; del buf44  # reuse
    # Source Nodes: [key_9], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg44_1, (2048, 2048), (1, 2048), 0), out=buf73)
    del arg44_1
    buf74 = reinterpret_tensor(buf37, (16, 128, 128), (16384, 128, 1), 0); del buf37  # reuse
    # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf72, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf73, (16, 128, 128), (128, 1, 2048), 0), out=buf74)
    buf75 = buf54; del buf54  # reuse
    buf76 = reinterpret_tensor(buf74, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf74  # reuse
    buf77 = buf52; del buf52  # reuse
    cpp_fused__softmax_lift_fresh_where_19(c_void_p(buf76.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()))
    del arg320_1
    buf78 = buf72; del buf72  # reuse
    # Source Nodes: [value_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg45_1, (2048, 2048), (1, 2048), 0), out=buf78)
    del arg45_1
    buf79 = buf76; del buf76  # reuse
    cpp_fused__softmax_20(c_void_p(buf79.data_ptr()), c_void_p(buf77.data_ptr()))
    buf80 = reinterpret_tensor(buf71, (16, 128, 128), (16384, 128, 1), 0); del buf71  # reuse
    # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf79, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf78, (16, 128, 128), (128, 2048, 1), 0), out=buf80)
    buf81 = reinterpret_tensor(buf79, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf79  # reuse
    cpp_fused_clone_21(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()))
    buf82 = reinterpret_tensor(buf80, (128, 2048), (2048, 1), 0); del buf80  # reuse
    # Source Nodes: [attn_output_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf81, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg46_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf82)
    del arg46_1
    del arg47_1
    buf83 = buf69; del buf69  # reuse
    buf84 = buf68; del buf68  # reuse
    buf86 = reinterpret_tensor(buf81, (1, 128, 2048), (262144, 2048, 1), 0); del buf81  # reuse
    cpp_fused_add_native_layer_norm_22(c_void_p(buf82.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()))
    del arg48_1
    del arg49_1
    buf87 = reinterpret_tensor(buf65, (128, 8192), (8192, 1), 0); del buf65  # reuse
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg51_1, reinterpret_tensor(buf86, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg50_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf87)
    del arg50_1
    del arg51_1
    buf88 = reinterpret_tensor(buf87, (1, 128, 8192), (1048576, 8192, 1), 0); del buf87  # reuse
    cpp_fused_add_mul_pow_tanh_23(c_void_p(buf88.data_ptr()))
    buf89 = reinterpret_tensor(buf86, (128, 2048), (2048, 1), 0); del buf86  # reuse
    # Source Nodes: [hidden_states_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf88, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg52_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf89)
    del arg52_1
    del arg53_1
    buf90 = buf84; del buf84  # reuse
    buf91 = buf83; del buf83  # reuse
    buf93 = buf22; del buf22  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf82.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    del arg54_1
    del arg55_1
    buf94 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg56_1, (2048, 2048), (1, 2048), 0), out=buf94)
    del arg56_1
    buf95 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg57_1, (2048, 2048), (1, 2048), 0), out=buf95)
    del arg57_1
    buf96 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf94, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf95, (16, 128, 128), (128, 1, 2048), 0), out=buf96)
    buf97 = buf77; del buf77  # reuse
    buf98 = reinterpret_tensor(buf96, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf96  # reuse
    buf99 = buf75; del buf75  # reuse
    cpp_fused__softmax_lift_fresh_where_25(c_void_p(buf98.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()))
    del arg321_1
    buf100 = buf94; del buf94  # reuse
    # Source Nodes: [value_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf93, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg58_1, (2048, 2048), (1, 2048), 0), out=buf100)
    del arg58_1
    buf101 = buf98; del buf98  # reuse
    cpp_fused__softmax_26(c_void_p(buf101.data_ptr()), c_void_p(buf99.data_ptr()))
    buf102 = reinterpret_tensor(buf93, (16, 128, 128), (16384, 128, 1), 0); del buf93  # reuse
    # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf101, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf100, (16, 128, 128), (128, 2048, 1), 0), out=buf102)
    buf103 = reinterpret_tensor(buf101, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf101  # reuse
    cpp_fused_clone_27(c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    buf104 = reinterpret_tensor(buf102, (128, 2048), (2048, 1), 0); del buf102  # reuse
    # Source Nodes: [attn_output_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf103, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg59_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf104)
    del arg59_1
    del arg60_1
    buf105 = buf91; del buf91  # reuse
    buf106 = buf90; del buf90  # reuse
    buf108 = reinterpret_tensor(buf103, (1, 128, 2048), (262144, 2048, 1), 0); del buf103  # reuse
    cpp_fused_add_native_layer_norm_28(c_void_p(buf104.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()))
    del arg61_1
    del arg62_1
    buf109 = reinterpret_tensor(buf88, (128, 8192), (8192, 1), 0); del buf88  # reuse
    # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf108, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg63_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf109)
    del arg63_1
    del arg64_1
    buf110 = reinterpret_tensor(buf109, (1, 128, 8192), (1048576, 8192, 1), 0); del buf109  # reuse
    cpp_fused_add_mul_pow_tanh_29(c_void_p(buf110.data_ptr()))
    buf111 = reinterpret_tensor(buf108, (128, 2048), (2048, 1), 0); del buf108  # reuse
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg66_1, reinterpret_tensor(buf110, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg65_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf111)
    del arg65_1
    del arg66_1
    buf112 = reinterpret_tensor(buf111, (1, 128, 2048), (262144, 2048, 1), 0); del buf111  # reuse
    buf113 = buf106; del buf106  # reuse
    buf114 = buf105; del buf105  # reuse
    buf116 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_30(c_void_p(buf112.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg67_1
    del arg68_1
    buf117 = buf89; del buf89  # reuse
    # Source Nodes: [query_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg69_1, (2048, 2048), (1, 2048), 0), out=buf117)
    del arg69_1
    buf118 = buf82; del buf82  # reuse
    # Source Nodes: [key_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg70_1, (2048, 2048), (1, 2048), 0), out=buf118)
    del arg70_1
    buf119 = reinterpret_tensor(buf67, (16, 128, 128), (16384, 128, 1), 0); del buf67  # reuse
    # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf117, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf118, (16, 128, 128), (128, 1, 2048), 0), out=buf119)
    buf120 = buf99; del buf99  # reuse
    buf121 = reinterpret_tensor(buf119, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf119  # reuse
    buf122 = buf97; del buf97  # reuse
    cpp_fused__softmax_lift_fresh_where_31(c_void_p(buf121.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()))
    del arg322_1
    buf123 = buf117; del buf117  # reuse
    # Source Nodes: [value_10], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg71_1, (2048, 2048), (1, 2048), 0), out=buf123)
    del arg71_1
    buf124 = buf121; del buf121  # reuse
    cpp_fused__softmax_32(c_void_p(buf124.data_ptr()), c_void_p(buf122.data_ptr()))
    buf125 = reinterpret_tensor(buf116, (16, 128, 128), (16384, 128, 1), 0); del buf116  # reuse
    # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf124, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf123, (16, 128, 128), (128, 2048, 1), 0), out=buf125)
    buf126 = reinterpret_tensor(buf124, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf124  # reuse
    cpp_fused_clone_33(c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    buf127 = reinterpret_tensor(buf125, (128, 2048), (2048, 1), 0); del buf125  # reuse
    # Source Nodes: [attn_output_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf126, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg72_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf127)
    del arg72_1
    del arg73_1
    buf128 = buf114; del buf114  # reuse
    buf129 = buf113; del buf113  # reuse
    buf131 = reinterpret_tensor(buf126, (1, 128, 2048), (262144, 2048, 1), 0); del buf126  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf127.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg74_1
    del arg75_1
    buf132 = reinterpret_tensor(buf110, (128, 8192), (8192, 1), 0); del buf110  # reuse
    # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg77_1, reinterpret_tensor(buf131, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg76_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf132)
    del arg76_1
    del arg77_1
    buf133 = reinterpret_tensor(buf132, (1, 128, 8192), (1048576, 8192, 1), 0); del buf132  # reuse
    cpp_fused_add_mul_pow_tanh_35(c_void_p(buf133.data_ptr()))
    buf134 = reinterpret_tensor(buf131, (128, 2048), (2048, 1), 0); del buf131  # reuse
    # Source Nodes: [hidden_states_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf133, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg78_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf134)
    del arg78_1
    del arg79_1
    buf135 = buf129; del buf129  # reuse
    buf136 = buf128; del buf128  # reuse
    buf138 = reinterpret_tensor(buf104, (1, 128, 2048), (262144, 2048, 1), 0); del buf104  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf127.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()))
    del arg80_1
    del arg81_1
    buf139 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg82_1, (2048, 2048), (1, 2048), 0), out=buf139)
    del arg82_1
    buf140 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg83_1, (2048, 2048), (1, 2048), 0), out=buf140)
    del arg83_1
    buf141 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf139, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf140, (16, 128, 128), (128, 1, 2048), 0), out=buf141)
    buf142 = buf122; del buf122  # reuse
    buf143 = reinterpret_tensor(buf141, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf141  # reuse
    buf144 = buf120; del buf120  # reuse
    cpp_fused__softmax_lift_fresh_where_37(c_void_p(buf143.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()))
    del arg323_1
    buf145 = buf139; del buf139  # reuse
    # Source Nodes: [value_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg84_1, (2048, 2048), (1, 2048), 0), out=buf145)
    del arg84_1
    buf146 = buf143; del buf143  # reuse
    cpp_fused__softmax_38(c_void_p(buf146.data_ptr()), c_void_p(buf144.data_ptr()))
    buf147 = reinterpret_tensor(buf138, (16, 128, 128), (16384, 128, 1), 0); del buf138  # reuse
    # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf146, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf145, (16, 128, 128), (128, 2048, 1), 0), out=buf147)
    buf148 = reinterpret_tensor(buf146, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf146  # reuse
    cpp_fused_clone_39(c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    buf149 = reinterpret_tensor(buf147, (128, 2048), (2048, 1), 0); del buf147  # reuse
    # Source Nodes: [attn_output_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf148, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg85_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf149)
    del arg85_1
    del arg86_1
    buf150 = buf136; del buf136  # reuse
    buf151 = buf135; del buf135  # reuse
    buf153 = reinterpret_tensor(buf148, (1, 128, 2048), (262144, 2048, 1), 0); del buf148  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf149.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg87_1
    del arg88_1
    buf154 = reinterpret_tensor(buf133, (128, 8192), (8192, 1), 0); del buf133  # reuse
    # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf153, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg89_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf154)
    del arg89_1
    del arg90_1
    buf155 = reinterpret_tensor(buf154, (1, 128, 8192), (1048576, 8192, 1), 0); del buf154  # reuse
    cpp_fused_add_mul_pow_tanh_41(c_void_p(buf155.data_ptr()))
    buf156 = reinterpret_tensor(buf153, (128, 2048), (2048, 1), 0); del buf153  # reuse
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg92_1, reinterpret_tensor(buf155, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg91_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf156)
    del arg91_1
    del arg92_1
    buf157 = reinterpret_tensor(buf156, (1, 128, 2048), (262144, 2048, 1), 0); del buf156  # reuse
    buf158 = buf151; del buf151  # reuse
    buf159 = buf150; del buf150  # reuse
    buf161 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_42(c_void_p(buf157.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    del arg93_1
    del arg94_1
    buf162 = buf149; del buf149  # reuse
    # Source Nodes: [query_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg95_1, (2048, 2048), (1, 2048), 0), out=buf162)
    del arg95_1
    buf163 = buf134; del buf134  # reuse
    # Source Nodes: [key_21], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg96_1, (2048, 2048), (1, 2048), 0), out=buf163)
    del arg96_1
    buf164 = reinterpret_tensor(buf127, (16, 128, 128), (16384, 128, 1), 0); del buf127  # reuse
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf162, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf163, (16, 128, 128), (128, 1, 2048), 0), out=buf164)
    buf165 = buf144; del buf144  # reuse
    buf166 = reinterpret_tensor(buf164, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf164  # reuse
    buf167 = buf142; del buf142  # reuse
    cpp_fused__softmax_lift_fresh_where_43(c_void_p(buf166.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg324_1
    buf168 = buf162; del buf162  # reuse
    # Source Nodes: [value_14], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf161, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg97_1, (2048, 2048), (1, 2048), 0), out=buf168)
    del arg97_1
    buf169 = buf166; del buf166  # reuse
    cpp_fused__softmax_44(c_void_p(buf169.data_ptr()), c_void_p(buf167.data_ptr()))
    buf170 = reinterpret_tensor(buf161, (16, 128, 128), (16384, 128, 1), 0); del buf161  # reuse
    # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf168, (16, 128, 128), (128, 2048, 1), 0), out=buf170)
    buf171 = reinterpret_tensor(buf169, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf169  # reuse
    cpp_fused_clone_45(c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    buf172 = reinterpret_tensor(buf170, (128, 2048), (2048, 1), 0); del buf170  # reuse
    # Source Nodes: [attn_output_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf171, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg98_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf172)
    del arg98_1
    del arg99_1
    buf173 = buf159; del buf159  # reuse
    buf174 = buf158; del buf158  # reuse
    buf176 = reinterpret_tensor(buf171, (1, 128, 2048), (262144, 2048, 1), 0); del buf171  # reuse
    cpp_fused_add_native_layer_norm_46(c_void_p(buf172.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf176.data_ptr()))
    del arg100_1
    del arg101_1
    buf177 = reinterpret_tensor(buf155, (128, 8192), (8192, 1), 0); del buf155  # reuse
    # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf176, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg102_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf177)
    del arg102_1
    del arg103_1
    buf178 = reinterpret_tensor(buf177, (1, 128, 8192), (1048576, 8192, 1), 0); del buf177  # reuse
    cpp_fused_add_mul_pow_tanh_47(c_void_p(buf178.data_ptr()))
    buf179 = reinterpret_tensor(buf176, (128, 2048), (2048, 1), 0); del buf176  # reuse
    # Source Nodes: [hidden_states_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf178, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg104_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf179)
    del arg104_1
    del arg105_1
    buf180 = buf174; del buf174  # reuse
    buf181 = buf173; del buf173  # reuse
    buf183 = buf112; del buf112  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf172.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    del arg106_1
    del arg107_1
    buf184 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_24], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg108_1, (2048, 2048), (1, 2048), 0), out=buf184)
    del arg108_1
    buf185 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_24], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg109_1, (2048, 2048), (1, 2048), 0), out=buf185)
    del arg109_1
    buf186 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf185, (16, 128, 128), (128, 1, 2048), 0), out=buf186)
    buf187 = buf167; del buf167  # reuse
    buf188 = reinterpret_tensor(buf186, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf186  # reuse
    buf189 = buf165; del buf165  # reuse
    cpp_fused__softmax_lift_fresh_where_49(c_void_p(buf188.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()))
    del arg325_1
    buf190 = buf184; del buf184  # reuse
    # Source Nodes: [value_16], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf183, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg110_1, (2048, 2048), (1, 2048), 0), out=buf190)
    del arg110_1
    buf191 = buf188; del buf188  # reuse
    cpp_fused__softmax_50(c_void_p(buf191.data_ptr()), c_void_p(buf189.data_ptr()))
    buf192 = reinterpret_tensor(buf183, (16, 128, 128), (16384, 128, 1), 0); del buf183  # reuse
    # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf191, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf190, (16, 128, 128), (128, 2048, 1), 0), out=buf192)
    buf193 = reinterpret_tensor(buf191, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf191  # reuse
    cpp_fused_clone_51(c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    buf194 = reinterpret_tensor(buf192, (128, 2048), (2048, 1), 0); del buf192  # reuse
    # Source Nodes: [attn_output_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg112_1, reinterpret_tensor(buf193, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg111_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf194)
    del arg111_1
    del arg112_1
    buf195 = buf181; del buf181  # reuse
    buf196 = buf180; del buf180  # reuse
    buf198 = reinterpret_tensor(buf193, (1, 128, 2048), (262144, 2048, 1), 0); del buf193  # reuse
    cpp_fused_add_native_layer_norm_52(c_void_p(buf194.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()))
    del arg113_1
    del arg114_1
    buf199 = reinterpret_tensor(buf178, (128, 8192), (8192, 1), 0); del buf178  # reuse
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg116_1, reinterpret_tensor(buf198, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg115_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf199)
    del arg115_1
    del arg116_1
    buf200 = reinterpret_tensor(buf199, (1, 128, 8192), (1048576, 8192, 1), 0); del buf199  # reuse
    cpp_fused_add_mul_pow_tanh_53(c_void_p(buf200.data_ptr()))
    buf201 = reinterpret_tensor(buf198, (128, 2048), (2048, 1), 0); del buf198  # reuse
    # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf200, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg117_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf201)
    del arg117_1
    del arg118_1
    buf202 = reinterpret_tensor(buf201, (1, 128, 2048), (262144, 2048, 1), 0); del buf201  # reuse
    buf203 = buf196; del buf196  # reuse
    buf204 = buf195; del buf195  # reuse
    buf206 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_54(c_void_p(buf202.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()))
    del arg119_1
    del arg120_1
    buf207 = buf194; del buf194  # reuse
    # Source Nodes: [query_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf206, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg121_1, (2048, 2048), (1, 2048), 0), out=buf207)
    del arg121_1
    buf208 = buf179; del buf179  # reuse
    # Source Nodes: [key_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf206, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg122_1, (2048, 2048), (1, 2048), 0), out=buf208)
    del arg122_1
    buf209 = reinterpret_tensor(buf172, (16, 128, 128), (16384, 128, 1), 0); del buf172  # reuse
    # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf207, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf208, (16, 128, 128), (128, 1, 2048), 0), out=buf209)
    buf210 = buf189; del buf189  # reuse
    buf211 = reinterpret_tensor(buf209, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf209  # reuse
    buf212 = buf187; del buf187  # reuse
    cpp_fused__softmax_lift_fresh_where_55(c_void_p(buf211.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()))
    del arg326_1
    buf213 = buf207; del buf207  # reuse
    # Source Nodes: [value_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf206, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg123_1, (2048, 2048), (1, 2048), 0), out=buf213)
    del arg123_1
    buf214 = buf211; del buf211  # reuse
    cpp_fused__softmax_56(c_void_p(buf214.data_ptr()), c_void_p(buf212.data_ptr()))
    buf215 = reinterpret_tensor(buf206, (16, 128, 128), (16384, 128, 1), 0); del buf206  # reuse
    # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf214, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf213, (16, 128, 128), (128, 2048, 1), 0), out=buf215)
    buf216 = reinterpret_tensor(buf214, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf214  # reuse
    cpp_fused_clone_57(c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    buf217 = reinterpret_tensor(buf215, (128, 2048), (2048, 1), 0); del buf215  # reuse
    # Source Nodes: [attn_output_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg125_1, reinterpret_tensor(buf216, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg124_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf217)
    del arg124_1
    del arg125_1
    buf218 = buf204; del buf204  # reuse
    buf219 = buf203; del buf203  # reuse
    buf221 = reinterpret_tensor(buf216, (1, 128, 2048), (262144, 2048, 1), 0); del buf216  # reuse
    cpp_fused_add_native_layer_norm_58(c_void_p(buf217.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()))
    del arg126_1
    del arg127_1
    buf222 = reinterpret_tensor(buf200, (128, 8192), (8192, 1), 0); del buf200  # reuse
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf221, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg128_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf222)
    del arg128_1
    del arg129_1
    buf223 = reinterpret_tensor(buf222, (1, 128, 8192), (1048576, 8192, 1), 0); del buf222  # reuse
    cpp_fused_add_mul_pow_tanh_59(c_void_p(buf223.data_ptr()))
    buf224 = reinterpret_tensor(buf221, (128, 2048), (2048, 1), 0); del buf221  # reuse
    # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf223, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg130_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf224)
    del arg130_1
    del arg131_1
    buf225 = buf219; del buf219  # reuse
    buf226 = buf218; del buf218  # reuse
    buf228 = buf157; del buf157  # reuse
    cpp_fused_add_native_layer_norm_60(c_void_p(buf217.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg132_1
    del arg133_1
    buf229 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg134_1, (2048, 2048), (1, 2048), 0), out=buf229)
    del arg134_1
    buf230 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg135_1, (2048, 2048), (1, 2048), 0), out=buf230)
    del arg135_1
    buf231 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf229, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf230, (16, 128, 128), (128, 1, 2048), 0), out=buf231)
    buf232 = buf212; del buf212  # reuse
    buf233 = reinterpret_tensor(buf231, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf231  # reuse
    buf234 = buf210; del buf210  # reuse
    cpp_fused__softmax_lift_fresh_where_61(c_void_p(buf233.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    del arg327_1
    buf235 = buf229; del buf229  # reuse
    # Source Nodes: [value_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg136_1, (2048, 2048), (1, 2048), 0), out=buf235)
    del arg136_1
    buf236 = buf233; del buf233  # reuse
    cpp_fused__softmax_62(c_void_p(buf236.data_ptr()), c_void_p(buf234.data_ptr()))
    buf237 = reinterpret_tensor(buf228, (16, 128, 128), (16384, 128, 1), 0); del buf228  # reuse
    # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf236, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf235, (16, 128, 128), (128, 2048, 1), 0), out=buf237)
    buf238 = reinterpret_tensor(buf236, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf236  # reuse
    cpp_fused_clone_63(c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    buf239 = reinterpret_tensor(buf237, (128, 2048), (2048, 1), 0); del buf237  # reuse
    # Source Nodes: [attn_output_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf238, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg137_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf239)
    del arg137_1
    del arg138_1
    buf240 = buf226; del buf226  # reuse
    buf241 = buf225; del buf225  # reuse
    buf243 = reinterpret_tensor(buf238, (1, 128, 2048), (262144, 2048, 1), 0); del buf238  # reuse
    cpp_fused_add_native_layer_norm_64(c_void_p(buf239.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()))
    del arg139_1
    del arg140_1
    buf244 = reinterpret_tensor(buf223, (128, 8192), (8192, 1), 0); del buf223  # reuse
    # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg142_1, reinterpret_tensor(buf243, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg141_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf244)
    del arg141_1
    del arg142_1
    buf245 = reinterpret_tensor(buf244, (1, 128, 8192), (1048576, 8192, 1), 0); del buf244  # reuse
    cpp_fused_add_mul_pow_tanh_65(c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf243, (128, 2048), (2048, 1), 0); del buf243  # reuse
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg144_1, reinterpret_tensor(buf245, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg143_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf246)
    del arg143_1
    del arg144_1
    buf247 = reinterpret_tensor(buf246, (1, 128, 2048), (262144, 2048, 1), 0); del buf246  # reuse
    buf248 = buf241; del buf241  # reuse
    buf249 = buf240; del buf240  # reuse
    buf251 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_66(c_void_p(buf247.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()))
    del arg145_1
    del arg146_1
    buf252 = buf239; del buf239  # reuse
    # Source Nodes: [query_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg147_1, (2048, 2048), (1, 2048), 0), out=buf252)
    del arg147_1
    buf253 = buf224; del buf224  # reuse
    # Source Nodes: [key_33], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg148_1, (2048, 2048), (1, 2048), 0), out=buf253)
    del arg148_1
    buf254 = reinterpret_tensor(buf217, (16, 128, 128), (16384, 128, 1), 0); del buf217  # reuse
    # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf252, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf253, (16, 128, 128), (128, 1, 2048), 0), out=buf254)
    buf255 = buf234; del buf234  # reuse
    buf256 = reinterpret_tensor(buf254, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf254  # reuse
    buf257 = buf232; del buf232  # reuse
    cpp_fused__softmax_lift_fresh_where_67(c_void_p(buf256.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del arg328_1
    buf258 = buf252; del buf252  # reuse
    # Source Nodes: [value_22], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg149_1, (2048, 2048), (1, 2048), 0), out=buf258)
    del arg149_1
    buf259 = buf256; del buf256  # reuse
    cpp_fused__softmax_68(c_void_p(buf259.data_ptr()), c_void_p(buf257.data_ptr()))
    buf260 = reinterpret_tensor(buf251, (16, 128, 128), (16384, 128, 1), 0); del buf251  # reuse
    # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf258, (16, 128, 128), (128, 2048, 1), 0), out=buf260)
    buf261 = reinterpret_tensor(buf259, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf259  # reuse
    cpp_fused_clone_69(c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    buf262 = reinterpret_tensor(buf260, (128, 2048), (2048, 1), 0); del buf260  # reuse
    # Source Nodes: [attn_output_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf261, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg150_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf262)
    del arg150_1
    del arg151_1
    buf263 = buf249; del buf249  # reuse
    buf264 = buf248; del buf248  # reuse
    buf266 = reinterpret_tensor(buf261, (1, 128, 2048), (262144, 2048, 1), 0); del buf261  # reuse
    cpp_fused_add_native_layer_norm_70(c_void_p(buf262.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del arg152_1
    del arg153_1
    buf267 = reinterpret_tensor(buf245, (128, 8192), (8192, 1), 0); del buf245  # reuse
    # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf266, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg154_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf267)
    del arg154_1
    del arg155_1
    buf268 = reinterpret_tensor(buf267, (1, 128, 8192), (1048576, 8192, 1), 0); del buf267  # reuse
    cpp_fused_add_mul_pow_tanh_71(c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf266, (128, 2048), (2048, 1), 0); del buf266  # reuse
    # Source Nodes: [hidden_states_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf268, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg156_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf269)
    del arg156_1
    del arg157_1
    buf270 = buf264; del buf264  # reuse
    buf271 = buf263; del buf263  # reuse
    buf273 = buf202; del buf202  # reuse
    cpp_fused_add_native_layer_norm_72(c_void_p(buf262.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()))
    del arg158_1
    del arg159_1
    buf274 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_36], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg160_1, (2048, 2048), (1, 2048), 0), out=buf274)
    del arg160_1
    buf275 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_36], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg161_1, (2048, 2048), (1, 2048), 0), out=buf275)
    del arg161_1
    buf276 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf274, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf275, (16, 128, 128), (128, 1, 2048), 0), out=buf276)
    buf277 = buf257; del buf257  # reuse
    buf278 = reinterpret_tensor(buf276, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf276  # reuse
    buf279 = buf255; del buf255  # reuse
    cpp_fused__softmax_lift_fresh_where_73(c_void_p(buf278.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()))
    del arg329_1
    buf280 = buf274; del buf274  # reuse
    # Source Nodes: [value_24], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg162_1, (2048, 2048), (1, 2048), 0), out=buf280)
    del arg162_1
    buf281 = buf278; del buf278  # reuse
    cpp_fused__softmax_74(c_void_p(buf281.data_ptr()), c_void_p(buf279.data_ptr()))
    buf282 = reinterpret_tensor(buf273, (16, 128, 128), (16384, 128, 1), 0); del buf273  # reuse
    # Source Nodes: [attn_output_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf281, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf280, (16, 128, 128), (128, 2048, 1), 0), out=buf282)
    buf283 = reinterpret_tensor(buf281, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf281  # reuse
    cpp_fused_clone_75(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    buf284 = reinterpret_tensor(buf282, (128, 2048), (2048, 1), 0); del buf282  # reuse
    # Source Nodes: [attn_output_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg164_1, reinterpret_tensor(buf283, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg163_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf284)
    del arg163_1
    del arg164_1
    buf285 = buf271; del buf271  # reuse
    buf286 = buf270; del buf270  # reuse
    buf288 = reinterpret_tensor(buf283, (1, 128, 2048), (262144, 2048, 1), 0); del buf283  # reuse
    cpp_fused_add_native_layer_norm_76(c_void_p(buf284.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()))
    del arg165_1
    del arg166_1
    buf289 = reinterpret_tensor(buf268, (128, 8192), (8192, 1), 0); del buf268  # reuse
    # Source Nodes: [hidden_states_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf288, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg167_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf289)
    del arg167_1
    del arg168_1
    buf290 = reinterpret_tensor(buf289, (1, 128, 8192), (1048576, 8192, 1), 0); del buf289  # reuse
    cpp_fused_add_mul_pow_tanh_77(c_void_p(buf290.data_ptr()))
    buf291 = reinterpret_tensor(buf288, (128, 2048), (2048, 1), 0); del buf288  # reuse
    # Source Nodes: [hidden_states_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf290, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg169_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf291)
    del arg169_1
    del arg170_1
    buf292 = reinterpret_tensor(buf291, (1, 128, 2048), (262144, 2048, 1), 0); del buf291  # reuse
    buf293 = buf286; del buf286  # reuse
    buf294 = buf285; del buf285  # reuse
    buf296 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_78(c_void_p(buf292.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()))
    del arg171_1
    del arg172_1
    buf297 = buf284; del buf284  # reuse
    # Source Nodes: [query_39], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg173_1, (2048, 2048), (1, 2048), 0), out=buf297)
    del arg173_1
    buf298 = buf269; del buf269  # reuse
    # Source Nodes: [key_39], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg174_1, (2048, 2048), (1, 2048), 0), out=buf298)
    del arg174_1
    buf299 = reinterpret_tensor(buf262, (16, 128, 128), (16384, 128, 1), 0); del buf262  # reuse
    # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf297, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf298, (16, 128, 128), (128, 1, 2048), 0), out=buf299)
    buf300 = buf279; del buf279  # reuse
    buf301 = reinterpret_tensor(buf299, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf299  # reuse
    buf302 = buf277; del buf277  # reuse
    cpp_fused__softmax_lift_fresh_where_79(c_void_p(buf301.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()))
    del arg330_1
    buf303 = buf297; del buf297  # reuse
    # Source Nodes: [value_26], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf296, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg175_1, (2048, 2048), (1, 2048), 0), out=buf303)
    del arg175_1
    buf304 = buf301; del buf301  # reuse
    cpp_fused__softmax_80(c_void_p(buf304.data_ptr()), c_void_p(buf302.data_ptr()))
    buf305 = reinterpret_tensor(buf296, (16, 128, 128), (16384, 128, 1), 0); del buf296  # reuse
    # Source Nodes: [attn_output_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf304, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf303, (16, 128, 128), (128, 2048, 1), 0), out=buf305)
    buf306 = reinterpret_tensor(buf304, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf304  # reuse
    cpp_fused_clone_81(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    buf307 = reinterpret_tensor(buf305, (128, 2048), (2048, 1), 0); del buf305  # reuse
    # Source Nodes: [attn_output_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf306, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg176_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf307)
    del arg176_1
    del arg177_1
    buf308 = buf294; del buf294  # reuse
    buf309 = buf293; del buf293  # reuse
    buf311 = reinterpret_tensor(buf306, (1, 128, 2048), (262144, 2048, 1), 0); del buf306  # reuse
    cpp_fused_add_native_layer_norm_82(c_void_p(buf307.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()))
    del arg178_1
    del arg179_1
    buf312 = reinterpret_tensor(buf290, (128, 8192), (8192, 1), 0); del buf290  # reuse
    # Source Nodes: [hidden_states_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf311, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg180_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf312)
    del arg180_1
    del arg181_1
    buf313 = reinterpret_tensor(buf312, (1, 128, 8192), (1048576, 8192, 1), 0); del buf312  # reuse
    cpp_fused_add_mul_pow_tanh_83(c_void_p(buf313.data_ptr()))
    buf314 = reinterpret_tensor(buf311, (128, 2048), (2048, 1), 0); del buf311  # reuse
    # Source Nodes: [hidden_states_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf313, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg182_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf314)
    del arg182_1
    del arg183_1
    buf315 = buf309; del buf309  # reuse
    buf316 = buf308; del buf308  # reuse
    buf318 = buf247; del buf247  # reuse
    cpp_fused_add_native_layer_norm_84(c_void_p(buf307.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()))
    del arg184_1
    del arg185_1
    buf319 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_42], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg186_1, (2048, 2048), (1, 2048), 0), out=buf319)
    del arg186_1
    buf320 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_42], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg187_1, (2048, 2048), (1, 2048), 0), out=buf320)
    del arg187_1
    buf321 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf319, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf320, (16, 128, 128), (128, 1, 2048), 0), out=buf321)
    buf322 = buf302; del buf302  # reuse
    buf323 = reinterpret_tensor(buf321, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf321  # reuse
    buf324 = buf300; del buf300  # reuse
    cpp_fused__softmax_lift_fresh_where_85(c_void_p(buf323.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()))
    del arg331_1
    buf325 = buf319; del buf319  # reuse
    # Source Nodes: [value_28], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf318, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg188_1, (2048, 2048), (1, 2048), 0), out=buf325)
    del arg188_1
    buf326 = buf323; del buf323  # reuse
    cpp_fused__softmax_86(c_void_p(buf326.data_ptr()), c_void_p(buf324.data_ptr()))
    buf327 = reinterpret_tensor(buf318, (16, 128, 128), (16384, 128, 1), 0); del buf318  # reuse
    # Source Nodes: [attn_output_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf326, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf325, (16, 128, 128), (128, 2048, 1), 0), out=buf327)
    buf328 = reinterpret_tensor(buf326, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf326  # reuse
    cpp_fused_clone_87(c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    buf329 = reinterpret_tensor(buf327, (128, 2048), (2048, 1), 0); del buf327  # reuse
    # Source Nodes: [attn_output_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf328, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg189_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf329)
    del arg189_1
    del arg190_1
    buf330 = buf316; del buf316  # reuse
    buf331 = buf315; del buf315  # reuse
    buf333 = reinterpret_tensor(buf328, (1, 128, 2048), (262144, 2048, 1), 0); del buf328  # reuse
    cpp_fused_add_native_layer_norm_88(c_void_p(buf329.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf333.data_ptr()))
    del arg191_1
    del arg192_1
    buf334 = reinterpret_tensor(buf313, (128, 8192), (8192, 1), 0); del buf313  # reuse
    # Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg194_1, reinterpret_tensor(buf333, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg193_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf334)
    del arg193_1
    del arg194_1
    buf335 = reinterpret_tensor(buf334, (1, 128, 8192), (1048576, 8192, 1), 0); del buf334  # reuse
    cpp_fused_add_mul_pow_tanh_89(c_void_p(buf335.data_ptr()))
    buf336 = reinterpret_tensor(buf333, (128, 2048), (2048, 1), 0); del buf333  # reuse
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf335, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg195_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf336)
    del arg195_1
    del arg196_1
    buf337 = reinterpret_tensor(buf336, (1, 128, 2048), (262144, 2048, 1), 0); del buf336  # reuse
    buf338 = buf331; del buf331  # reuse
    buf339 = buf330; del buf330  # reuse
    buf341 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_90(c_void_p(buf337.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()))
    del arg197_1
    del arg198_1
    buf342 = buf329; del buf329  # reuse
    # Source Nodes: [query_45], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg199_1, (2048, 2048), (1, 2048), 0), out=buf342)
    del arg199_1
    buf343 = buf314; del buf314  # reuse
    # Source Nodes: [key_45], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg200_1, (2048, 2048), (1, 2048), 0), out=buf343)
    del arg200_1
    buf344 = reinterpret_tensor(buf307, (16, 128, 128), (16384, 128, 1), 0); del buf307  # reuse
    # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf342, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf343, (16, 128, 128), (128, 1, 2048), 0), out=buf344)
    buf345 = buf324; del buf324  # reuse
    buf346 = reinterpret_tensor(buf344, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf344  # reuse
    buf347 = buf322; del buf322  # reuse
    cpp_fused__softmax_lift_fresh_where_91(c_void_p(buf346.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    del arg332_1
    buf348 = buf342; del buf342  # reuse
    # Source Nodes: [value_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf341, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg201_1, (2048, 2048), (1, 2048), 0), out=buf348)
    del arg201_1
    buf349 = buf346; del buf346  # reuse
    cpp_fused__softmax_92(c_void_p(buf349.data_ptr()), c_void_p(buf347.data_ptr()))
    buf350 = reinterpret_tensor(buf341, (16, 128, 128), (16384, 128, 1), 0); del buf341  # reuse
    # Source Nodes: [attn_output_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf349, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf348, (16, 128, 128), (128, 2048, 1), 0), out=buf350)
    buf351 = reinterpret_tensor(buf349, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf349  # reuse
    cpp_fused_clone_93(c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    buf352 = reinterpret_tensor(buf350, (128, 2048), (2048, 1), 0); del buf350  # reuse
    # Source Nodes: [attn_output_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg203_1, reinterpret_tensor(buf351, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg202_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf352)
    del arg202_1
    del arg203_1
    buf353 = buf339; del buf339  # reuse
    buf354 = buf338; del buf338  # reuse
    buf356 = reinterpret_tensor(buf351, (1, 128, 2048), (262144, 2048, 1), 0); del buf351  # reuse
    cpp_fused_add_native_layer_norm_94(c_void_p(buf352.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf356.data_ptr()))
    del arg204_1
    del arg205_1
    buf357 = reinterpret_tensor(buf335, (128, 8192), (8192, 1), 0); del buf335  # reuse
    # Source Nodes: [hidden_states_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg207_1, reinterpret_tensor(buf356, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg206_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf357)
    del arg206_1
    del arg207_1
    buf358 = reinterpret_tensor(buf357, (1, 128, 8192), (1048576, 8192, 1), 0); del buf357  # reuse
    cpp_fused_add_mul_pow_tanh_95(c_void_p(buf358.data_ptr()))
    buf359 = reinterpret_tensor(buf356, (128, 2048), (2048, 1), 0); del buf356  # reuse
    # Source Nodes: [hidden_states_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg209_1, reinterpret_tensor(buf358, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg208_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf359)
    del arg208_1
    del arg209_1
    buf360 = buf354; del buf354  # reuse
    buf361 = buf353; del buf353  # reuse
    buf363 = buf292; del buf292  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf352.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()))
    del arg210_1
    del arg211_1
    buf364 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_48], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg212_1, (2048, 2048), (1, 2048), 0), out=buf364)
    del arg212_1
    buf365 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_48], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg213_1, (2048, 2048), (1, 2048), 0), out=buf365)
    del arg213_1
    buf366 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_96], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf364, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf365, (16, 128, 128), (128, 1, 2048), 0), out=buf366)
    buf367 = buf347; del buf347  # reuse
    buf368 = reinterpret_tensor(buf366, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf366  # reuse
    buf369 = buf345; del buf345  # reuse
    cpp_fused__softmax_lift_fresh_where_97(c_void_p(buf368.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()))
    del arg333_1
    buf370 = buf364; del buf364  # reuse
    # Source Nodes: [value_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg214_1, (2048, 2048), (1, 2048), 0), out=buf370)
    del arg214_1
    buf371 = buf368; del buf368  # reuse
    cpp_fused__softmax_98(c_void_p(buf371.data_ptr()), c_void_p(buf369.data_ptr()))
    buf372 = reinterpret_tensor(buf363, (16, 128, 128), (16384, 128, 1), 0); del buf363  # reuse
    # Source Nodes: [attn_output_96], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf371, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf370, (16, 128, 128), (128, 2048, 1), 0), out=buf372)
    buf373 = reinterpret_tensor(buf371, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf371  # reuse
    cpp_fused_clone_99(c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()))
    buf374 = reinterpret_tensor(buf372, (128, 2048), (2048, 1), 0); del buf372  # reuse
    # Source Nodes: [attn_output_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg216_1, reinterpret_tensor(buf373, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg215_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf374)
    del arg215_1
    del arg216_1
    buf375 = buf361; del buf361  # reuse
    buf376 = buf360; del buf360  # reuse
    buf378 = reinterpret_tensor(buf373, (1, 128, 2048), (262144, 2048, 1), 0); del buf373  # reuse
    cpp_fused_add_native_layer_norm_100(c_void_p(buf374.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()))
    del arg217_1
    del arg218_1
    buf379 = reinterpret_tensor(buf358, (128, 8192), (8192, 1), 0); del buf358  # reuse
    # Source Nodes: [hidden_states_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf378, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg219_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf379)
    del arg219_1
    del arg220_1
    buf380 = reinterpret_tensor(buf379, (1, 128, 8192), (1048576, 8192, 1), 0); del buf379  # reuse
    cpp_fused_add_mul_pow_tanh_101(c_void_p(buf380.data_ptr()))
    buf381 = reinterpret_tensor(buf378, (128, 2048), (2048, 1), 0); del buf378  # reuse
    # Source Nodes: [hidden_states_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg222_1, reinterpret_tensor(buf380, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg221_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf381)
    del arg221_1
    del arg222_1
    buf382 = reinterpret_tensor(buf381, (1, 128, 2048), (262144, 2048, 1), 0); del buf381  # reuse
    buf383 = buf376; del buf376  # reuse
    buf384 = buf375; del buf375  # reuse
    buf386 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_102(c_void_p(buf382.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf386.data_ptr()))
    del arg223_1
    del arg224_1
    buf387 = buf374; del buf374  # reuse
    # Source Nodes: [query_51], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg225_1, (2048, 2048), (1, 2048), 0), out=buf387)
    del arg225_1
    buf388 = buf359; del buf359  # reuse
    # Source Nodes: [key_51], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg226_1, (2048, 2048), (1, 2048), 0), out=buf388)
    del arg226_1
    buf389 = reinterpret_tensor(buf352, (16, 128, 128), (16384, 128, 1), 0); del buf352  # reuse
    # Source Nodes: [attn_weights_102], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf387, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf388, (16, 128, 128), (128, 1, 2048), 0), out=buf389)
    buf390 = buf369; del buf369  # reuse
    buf391 = reinterpret_tensor(buf389, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf389  # reuse
    buf392 = buf367; del buf367  # reuse
    cpp_fused__softmax_lift_fresh_where_103(c_void_p(buf391.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf392.data_ptr()))
    del arg334_1
    buf393 = buf387; del buf387  # reuse
    # Source Nodes: [value_34], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf386, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg227_1, (2048, 2048), (1, 2048), 0), out=buf393)
    del arg227_1
    buf394 = buf391; del buf391  # reuse
    cpp_fused__softmax_104(c_void_p(buf394.data_ptr()), c_void_p(buf392.data_ptr()))
    buf395 = reinterpret_tensor(buf386, (16, 128, 128), (16384, 128, 1), 0); del buf386  # reuse
    # Source Nodes: [attn_output_102], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf394, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf393, (16, 128, 128), (128, 2048, 1), 0), out=buf395)
    buf396 = reinterpret_tensor(buf394, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf394  # reuse
    cpp_fused_clone_105(c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    buf397 = reinterpret_tensor(buf395, (128, 2048), (2048, 1), 0); del buf395  # reuse
    # Source Nodes: [attn_output_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg229_1, reinterpret_tensor(buf396, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg228_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf397)
    del arg228_1
    del arg229_1
    buf398 = buf384; del buf384  # reuse
    buf399 = buf383; del buf383  # reuse
    buf401 = reinterpret_tensor(buf396, (1, 128, 2048), (262144, 2048, 1), 0); del buf396  # reuse
    cpp_fused_add_native_layer_norm_106(c_void_p(buf397.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()))
    del arg230_1
    del arg231_1
    buf402 = reinterpret_tensor(buf380, (128, 8192), (8192, 1), 0); del buf380  # reuse
    # Source Nodes: [hidden_states_158], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg233_1, reinterpret_tensor(buf401, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg232_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf402)
    del arg232_1
    del arg233_1
    buf403 = reinterpret_tensor(buf402, (1, 128, 8192), (1048576, 8192, 1), 0); del buf402  # reuse
    cpp_fused_add_mul_pow_tanh_107(c_void_p(buf403.data_ptr()))
    buf404 = reinterpret_tensor(buf401, (128, 2048), (2048, 1), 0); del buf401  # reuse
    # Source Nodes: [hidden_states_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg235_1, reinterpret_tensor(buf403, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg234_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf404)
    del arg234_1
    del arg235_1
    buf405 = buf399; del buf399  # reuse
    buf406 = buf398; del buf398  # reuse
    buf408 = buf337; del buf337  # reuse
    cpp_fused_add_native_layer_norm_108(c_void_p(buf397.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf408.data_ptr()))
    del arg236_1
    del arg237_1
    buf409 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_54], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg238_1, (2048, 2048), (1, 2048), 0), out=buf409)
    del arg238_1
    buf410 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_54], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg239_1, (2048, 2048), (1, 2048), 0), out=buf410)
    del arg239_1
    buf411 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_108], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf409, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf410, (16, 128, 128), (128, 1, 2048), 0), out=buf411)
    buf412 = buf392; del buf392  # reuse
    buf413 = reinterpret_tensor(buf411, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf411  # reuse
    buf414 = buf390; del buf390  # reuse
    cpp_fused__softmax_lift_fresh_where_109(c_void_p(buf413.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf414.data_ptr()))
    del arg335_1
    buf415 = buf409; del buf409  # reuse
    # Source Nodes: [value_36], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf408, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg240_1, (2048, 2048), (1, 2048), 0), out=buf415)
    del arg240_1
    buf416 = buf413; del buf413  # reuse
    cpp_fused__softmax_110(c_void_p(buf416.data_ptr()), c_void_p(buf414.data_ptr()))
    buf417 = reinterpret_tensor(buf408, (16, 128, 128), (16384, 128, 1), 0); del buf408  # reuse
    # Source Nodes: [attn_output_108], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf416, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf415, (16, 128, 128), (128, 2048, 1), 0), out=buf417)
    buf418 = reinterpret_tensor(buf416, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf416  # reuse
    cpp_fused_clone_111(c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    buf419 = reinterpret_tensor(buf417, (128, 2048), (2048, 1), 0); del buf417  # reuse
    # Source Nodes: [attn_output_110], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg242_1, reinterpret_tensor(buf418, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg241_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf419)
    del arg241_1
    del arg242_1
    buf420 = buf406; del buf406  # reuse
    buf421 = buf405; del buf405  # reuse
    buf423 = reinterpret_tensor(buf418, (1, 128, 2048), (262144, 2048, 1), 0); del buf418  # reuse
    cpp_fused_add_native_layer_norm_112(c_void_p(buf419.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()))
    del arg243_1
    del arg244_1
    buf424 = reinterpret_tensor(buf403, (128, 8192), (8192, 1), 0); del buf403  # reuse
    # Source Nodes: [hidden_states_167], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg246_1, reinterpret_tensor(buf423, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg245_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf424)
    del arg245_1
    del arg246_1
    buf425 = reinterpret_tensor(buf424, (1, 128, 8192), (1048576, 8192, 1), 0); del buf424  # reuse
    cpp_fused_add_mul_pow_tanh_113(c_void_p(buf425.data_ptr()))
    buf426 = reinterpret_tensor(buf423, (128, 2048), (2048, 1), 0); del buf423  # reuse
    # Source Nodes: [hidden_states_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg248_1, reinterpret_tensor(buf425, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg247_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf426)
    del arg247_1
    del arg248_1
    buf427 = reinterpret_tensor(buf426, (1, 128, 2048), (262144, 2048, 1), 0); del buf426  # reuse
    buf428 = buf421; del buf421  # reuse
    buf429 = buf420; del buf420  # reuse
    buf431 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_114(c_void_p(buf427.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()))
    del arg249_1
    del arg250_1
    buf432 = buf419; del buf419  # reuse
    # Source Nodes: [query_57], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg251_1, (2048, 2048), (1, 2048), 0), out=buf432)
    del arg251_1
    buf433 = buf404; del buf404  # reuse
    # Source Nodes: [key_57], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg252_1, (2048, 2048), (1, 2048), 0), out=buf433)
    del arg252_1
    buf434 = reinterpret_tensor(buf397, (16, 128, 128), (16384, 128, 1), 0); del buf397  # reuse
    # Source Nodes: [attn_weights_114], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf432, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf433, (16, 128, 128), (128, 1, 2048), 0), out=buf434)
    buf435 = buf414; del buf414  # reuse
    buf436 = reinterpret_tensor(buf434, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf434  # reuse
    buf437 = buf412; del buf412  # reuse
    cpp_fused__softmax_lift_fresh_where_115(c_void_p(buf436.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf437.data_ptr()))
    del arg336_1
    buf438 = buf432; del buf432  # reuse
    # Source Nodes: [value_38], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf431, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg253_1, (2048, 2048), (1, 2048), 0), out=buf438)
    del arg253_1
    buf439 = buf436; del buf436  # reuse
    cpp_fused__softmax_116(c_void_p(buf439.data_ptr()), c_void_p(buf437.data_ptr()))
    buf440 = reinterpret_tensor(buf431, (16, 128, 128), (16384, 128, 1), 0); del buf431  # reuse
    # Source Nodes: [attn_output_114], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf439, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf438, (16, 128, 128), (128, 2048, 1), 0), out=buf440)
    buf441 = reinterpret_tensor(buf439, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf439  # reuse
    cpp_fused_clone_117(c_void_p(buf440.data_ptr()), c_void_p(buf441.data_ptr()))
    buf442 = reinterpret_tensor(buf440, (128, 2048), (2048, 1), 0); del buf440  # reuse
    # Source Nodes: [attn_output_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg255_1, reinterpret_tensor(buf441, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg254_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf442)
    del arg254_1
    del arg255_1
    buf443 = buf429; del buf429  # reuse
    buf444 = buf428; del buf428  # reuse
    buf446 = reinterpret_tensor(buf441, (1, 128, 2048), (262144, 2048, 1), 0); del buf441  # reuse
    cpp_fused_add_native_layer_norm_118(c_void_p(buf442.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf446.data_ptr()))
    del arg256_1
    del arg257_1
    buf447 = reinterpret_tensor(buf425, (128, 8192), (8192, 1), 0); del buf425  # reuse
    # Source Nodes: [hidden_states_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg259_1, reinterpret_tensor(buf446, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg258_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf447)
    del arg258_1
    del arg259_1
    buf448 = reinterpret_tensor(buf447, (1, 128, 8192), (1048576, 8192, 1), 0); del buf447  # reuse
    cpp_fused_add_mul_pow_tanh_119(c_void_p(buf448.data_ptr()))
    buf449 = reinterpret_tensor(buf446, (128, 2048), (2048, 1), 0); del buf446  # reuse
    # Source Nodes: [hidden_states_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg261_1, reinterpret_tensor(buf448, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg260_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf449)
    del arg260_1
    del arg261_1
    buf450 = buf444; del buf444  # reuse
    buf451 = buf443; del buf443  # reuse
    buf453 = buf382; del buf382  # reuse
    cpp_fused_add_native_layer_norm_120(c_void_p(buf442.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf453.data_ptr()))
    del arg262_1
    del arg263_1
    buf454 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg264_1, (2048, 2048), (1, 2048), 0), out=buf454)
    del arg264_1
    buf455 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg265_1, (2048, 2048), (1, 2048), 0), out=buf455)
    del arg265_1
    buf456 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf454, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf455, (16, 128, 128), (128, 1, 2048), 0), out=buf456)
    buf457 = buf437; del buf437  # reuse
    buf458 = reinterpret_tensor(buf456, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf456  # reuse
    buf459 = buf435; del buf435  # reuse
    cpp_fused__softmax_lift_fresh_where_121(c_void_p(buf458.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf459.data_ptr()))
    del arg337_1
    buf460 = buf454; del buf454  # reuse
    # Source Nodes: [value_40], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf453, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg266_1, (2048, 2048), (1, 2048), 0), out=buf460)
    del arg266_1
    buf461 = buf458; del buf458  # reuse
    cpp_fused__softmax_122(c_void_p(buf461.data_ptr()), c_void_p(buf459.data_ptr()))
    buf462 = reinterpret_tensor(buf453, (16, 128, 128), (16384, 128, 1), 0); del buf453  # reuse
    # Source Nodes: [attn_output_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf461, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf460, (16, 128, 128), (128, 2048, 1), 0), out=buf462)
    buf463 = reinterpret_tensor(buf461, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf461  # reuse
    cpp_fused_clone_123(c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()))
    buf464 = reinterpret_tensor(buf462, (128, 2048), (2048, 1), 0); del buf462  # reuse
    # Source Nodes: [attn_output_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf463, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg267_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf464)
    del arg267_1
    del arg268_1
    buf465 = buf451; del buf451  # reuse
    buf466 = buf450; del buf450  # reuse
    buf468 = reinterpret_tensor(buf463, (1, 128, 2048), (262144, 2048, 1), 0); del buf463  # reuse
    cpp_fused_add_native_layer_norm_124(c_void_p(buf464.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()))
    del arg269_1
    del arg270_1
    buf469 = reinterpret_tensor(buf448, (128, 8192), (8192, 1), 0); del buf448  # reuse
    # Source Nodes: [hidden_states_185], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg272_1, reinterpret_tensor(buf468, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg271_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf469)
    del arg271_1
    del arg272_1
    buf470 = reinterpret_tensor(buf469, (1, 128, 8192), (1048576, 8192, 1), 0); del buf469  # reuse
    cpp_fused_add_mul_pow_tanh_125(c_void_p(buf470.data_ptr()))
    buf471 = reinterpret_tensor(buf468, (128, 2048), (2048, 1), 0); del buf468  # reuse
    # Source Nodes: [hidden_states_187], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg274_1, reinterpret_tensor(buf470, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg273_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf471)
    del arg273_1
    del arg274_1
    buf472 = reinterpret_tensor(buf471, (1, 128, 2048), (262144, 2048, 1), 0); del buf471  # reuse
    buf473 = buf466; del buf466  # reuse
    buf474 = buf465; del buf465  # reuse
    buf476 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_126(c_void_p(buf472.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()))
    del arg275_1
    del arg276_1
    buf477 = buf464; del buf464  # reuse
    # Source Nodes: [query_63], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg277_1, (2048, 2048), (1, 2048), 0), out=buf477)
    del arg277_1
    buf478 = buf449; del buf449  # reuse
    # Source Nodes: [key_63], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg278_1, (2048, 2048), (1, 2048), 0), out=buf478)
    del arg278_1
    buf479 = reinterpret_tensor(buf442, (16, 128, 128), (16384, 128, 1), 0); del buf442  # reuse
    # Source Nodes: [attn_weights_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf477, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf478, (16, 128, 128), (128, 1, 2048), 0), out=buf479)
    buf480 = buf459; del buf459  # reuse
    buf481 = reinterpret_tensor(buf479, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf479  # reuse
    buf482 = buf457; del buf457  # reuse
    cpp_fused__softmax_lift_fresh_where_127(c_void_p(buf481.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf482.data_ptr()))
    del arg338_1
    buf483 = buf477; del buf477  # reuse
    # Source Nodes: [value_42], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf476, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg279_1, (2048, 2048), (1, 2048), 0), out=buf483)
    del arg279_1
    buf484 = buf481; del buf481  # reuse
    cpp_fused__softmax_128(c_void_p(buf484.data_ptr()), c_void_p(buf482.data_ptr()))
    buf485 = reinterpret_tensor(buf476, (16, 128, 128), (16384, 128, 1), 0); del buf476  # reuse
    # Source Nodes: [attn_output_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf484, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf483, (16, 128, 128), (128, 2048, 1), 0), out=buf485)
    buf486 = reinterpret_tensor(buf484, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf484  # reuse
    cpp_fused_clone_129(c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    buf487 = reinterpret_tensor(buf485, (128, 2048), (2048, 1), 0); del buf485  # reuse
    # Source Nodes: [attn_output_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg281_1, reinterpret_tensor(buf486, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg280_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf487)
    del arg280_1
    del arg281_1
    buf488 = buf474; del buf474  # reuse
    buf489 = buf473; del buf473  # reuse
    buf491 = reinterpret_tensor(buf486, (1, 128, 2048), (262144, 2048, 1), 0); del buf486  # reuse
    cpp_fused_add_native_layer_norm_130(c_void_p(buf487.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf491.data_ptr()))
    del arg282_1
    del arg283_1
    buf492 = reinterpret_tensor(buf470, (128, 8192), (8192, 1), 0); del buf470  # reuse
    # Source Nodes: [hidden_states_194], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg285_1, reinterpret_tensor(buf491, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg284_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf492)
    del arg284_1
    del arg285_1
    buf493 = reinterpret_tensor(buf492, (1, 128, 8192), (1048576, 8192, 1), 0); del buf492  # reuse
    cpp_fused_add_mul_pow_tanh_131(c_void_p(buf493.data_ptr()))
    buf494 = reinterpret_tensor(buf491, (128, 2048), (2048, 1), 0); del buf491  # reuse
    # Source Nodes: [hidden_states_196], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg287_1, reinterpret_tensor(buf493, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg286_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf494)
    del arg286_1
    del arg287_1
    buf495 = buf489; del buf489  # reuse
    buf496 = buf488; del buf488  # reuse
    buf498 = buf427; del buf427  # reuse
    cpp_fused_add_native_layer_norm_132(c_void_p(buf487.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf498.data_ptr()))
    del arg288_1
    del arg289_1
    buf499 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_66], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf498, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg290_1, (2048, 2048), (1, 2048), 0), out=buf499)
    del arg290_1
    buf500 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_66], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf498, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg291_1, (2048, 2048), (1, 2048), 0), out=buf500)
    del arg291_1
    buf501 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_132], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf499, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf500, (16, 128, 128), (128, 1, 2048), 0), out=buf501)
    buf502 = buf482; del buf482  # reuse
    buf503 = reinterpret_tensor(buf501, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf501  # reuse
    buf504 = buf480; del buf480  # reuse
    cpp_fused__softmax_lift_fresh_where_133(c_void_p(buf503.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf504.data_ptr()))
    del arg339_1
    buf505 = buf499; del buf499  # reuse
    # Source Nodes: [value_44], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf498, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg292_1, (2048, 2048), (1, 2048), 0), out=buf505)
    del arg292_1
    buf506 = buf503; del buf503  # reuse
    cpp_fused__softmax_134(c_void_p(buf506.data_ptr()), c_void_p(buf504.data_ptr()))
    buf507 = reinterpret_tensor(buf498, (16, 128, 128), (16384, 128, 1), 0); del buf498  # reuse
    # Source Nodes: [attn_output_132], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf506, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf505, (16, 128, 128), (128, 2048, 1), 0), out=buf507)
    buf508 = reinterpret_tensor(buf506, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf506  # reuse
    cpp_fused_clone_135(c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()))
    buf509 = reinterpret_tensor(buf507, (128, 2048), (2048, 1), 0); del buf507  # reuse
    # Source Nodes: [attn_output_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg294_1, reinterpret_tensor(buf508, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg293_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf509)
    del arg293_1
    del arg294_1
    buf510 = buf496; del buf496  # reuse
    buf511 = buf495; del buf495  # reuse
    buf513 = reinterpret_tensor(buf508, (1, 128, 2048), (262144, 2048, 1), 0); del buf508  # reuse
    cpp_fused_add_native_layer_norm_136(c_void_p(buf509.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf513.data_ptr()))
    del arg295_1
    del arg296_1
    buf514 = reinterpret_tensor(buf493, (128, 8192), (8192, 1), 0); del buf493  # reuse
    # Source Nodes: [hidden_states_203], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg298_1, reinterpret_tensor(buf513, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg297_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf514)
    del arg297_1
    del arg298_1
    buf515 = reinterpret_tensor(buf514, (1, 128, 8192), (1048576, 8192, 1), 0); del buf514  # reuse
    cpp_fused_add_mul_pow_tanh_137(c_void_p(buf515.data_ptr()))
    buf516 = reinterpret_tensor(buf513, (128, 2048), (2048, 1), 0); del buf513  # reuse
    # Source Nodes: [hidden_states_205], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg300_1, reinterpret_tensor(buf515, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg299_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf516)
    del arg299_1
    del arg300_1
    buf517 = reinterpret_tensor(buf516, (1, 128, 2048), (262144, 2048, 1), 0); del buf516  # reuse
    buf518 = buf511; del buf511  # reuse
    buf519 = buf510; del buf510  # reuse
    buf521 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_138(c_void_p(buf517.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf521.data_ptr()))
    del arg301_1
    del arg302_1
    buf522 = buf509; del buf509  # reuse
    # Source Nodes: [query_69], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg303_1, (2048, 2048), (1, 2048), 0), out=buf522)
    del arg303_1
    buf523 = buf494; del buf494  # reuse
    # Source Nodes: [key_69], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg304_1, (2048, 2048), (1, 2048), 0), out=buf523)
    del arg304_1
    buf524 = reinterpret_tensor(buf487, (16, 128, 128), (16384, 128, 1), 0); del buf487  # reuse
    # Source Nodes: [attn_weights_138], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf522, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf523, (16, 128, 128), (128, 1, 2048), 0), out=buf524)
    buf525 = buf504; del buf504  # reuse
    buf526 = reinterpret_tensor(buf524, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf524  # reuse
    buf527 = buf502; del buf502  # reuse
    cpp_fused__softmax_lift_fresh_where_139(c_void_p(buf526.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf527.data_ptr()))
    del arg340_1
    del buf525
    buf528 = buf522; del buf522  # reuse
    # Source Nodes: [value_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf521, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg305_1, (2048, 2048), (1, 2048), 0), out=buf528)
    del arg305_1
    buf529 = buf526; del buf526  # reuse
    cpp_fused__softmax_140(c_void_p(buf529.data_ptr()), c_void_p(buf527.data_ptr()))
    del buf527
    buf530 = reinterpret_tensor(buf521, (16, 128, 128), (16384, 128, 1), 0); del buf521  # reuse
    # Source Nodes: [attn_output_138], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf529, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf528, (16, 128, 128), (128, 2048, 1), 0), out=buf530)
    buf531 = reinterpret_tensor(buf529, (1, 128, 16, 128), (262144, 2048, 128, 1), 0); del buf529  # reuse
    cpp_fused_clone_141(c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()))
    buf532 = reinterpret_tensor(buf530, (128, 2048), (2048, 1), 0); del buf530  # reuse
    # Source Nodes: [attn_output_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg307_1, reinterpret_tensor(buf531, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg306_1, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf532)
    del arg306_1
    del arg307_1
    buf533 = buf519; del buf519  # reuse
    buf534 = buf518; del buf518  # reuse
    buf536 = reinterpret_tensor(buf531, (1, 128, 2048), (262144, 2048, 1), 0); del buf531  # reuse
    cpp_fused_add_native_layer_norm_142(c_void_p(buf532.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf536.data_ptr()))
    del arg308_1
    del arg309_1
    buf537 = reinterpret_tensor(buf515, (128, 8192), (8192, 1), 0); del buf515  # reuse
    # Source Nodes: [hidden_states_212], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg311_1, reinterpret_tensor(buf536, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg310_1, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf537)
    del arg310_1
    del arg311_1
    buf538 = reinterpret_tensor(buf537, (1, 128, 8192), (1048576, 8192, 1), 0); del buf537  # reuse
    cpp_fused_add_mul_pow_tanh_143(c_void_p(buf538.data_ptr()))
    buf539 = reinterpret_tensor(buf536, (128, 2048), (2048, 1), 0); del buf536  # reuse
    # Source Nodes: [hidden_states_214], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg313_1, reinterpret_tensor(buf538, (128, 8192), (8192, 1), 0), reinterpret_tensor(arg312_1, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf539)
    del arg312_1
    del arg313_1
    del buf538
    buf540 = buf534; del buf534  # reuse
    buf541 = buf533; del buf533  # reuse
    buf543 = buf472; del buf472  # reuse
    cpp_fused_add_native_layer_norm_144(c_void_p(buf532.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf543.data_ptr()))
    del arg314_1
    del arg315_1
    del buf517
    del buf532
    del buf539
    del buf540
    del buf541
    buf544 = empty((128, 50257), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf543, (128, 2048), (2048, 1), 0), reinterpret_tensor(arg316_1, (2048, 50257), (1, 2048), 0), out=buf544)
    del arg316_1
    del buf543
    buf545 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.float32)
    buf546 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.float32)
    buf547 = empty((), device='cpu', dtype=torch.float32)
    buf548 = empty((), device='cpu', dtype=torch.int64)
    buf549 = buf547; del buf547  # reuse
    cpp_fused__log_softmax_nll_loss_forward_145(c_void_p(buf549.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf548.data_ptr()))
    del arg342_1
    return (buf549, reinterpret_tensor(buf544, (1, 128, 50257), (6432896, 50257, 1), 0), reinterpret_tensor(buf5, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf10, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf28, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf33, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf50, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf55, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf73, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf78, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf95, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf100, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf118, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf123, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf140, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf145, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf163, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf168, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf185, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf190, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf208, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf213, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf230, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf235, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf253, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf258, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf275, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf280, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf298, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf303, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf320, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf325, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf343, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf348, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf365, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf370, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf388, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf393, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf410, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf415, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf433, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf438, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf455, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf460, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf478, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf483, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf500, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf505, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf523, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf528, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((50257, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((50257, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg318_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg319_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg320_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg321_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg322_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg323_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg324_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg325_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg326_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg327_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg328_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg329_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg330_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg331_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg332_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg333_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg334_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg335_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg336_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg337_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg338_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg339_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg340_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg341_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg342_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPTNeoForCausalLM', benchmark_compiled_module)
