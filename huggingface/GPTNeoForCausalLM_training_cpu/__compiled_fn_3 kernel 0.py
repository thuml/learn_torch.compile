
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


cpp_fused_add_embedding_native_layer_norm_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = c10::convert<long>(x0);
            out_ptr0[static_cast<long>(x0)] = tmp0;
        }
    }
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
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
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
                    auto tmp7 = out_ptr1[static_cast<long>(x0)];
                    auto tmp10 = out_ptr2[static_cast<long>(x0)];
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
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr4 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_embedding_native_layer_norm_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const long* in_ptr1,
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_embedding_native_layer_norm_view_5 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_10 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_13 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_15 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_17 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_22 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_23 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_25 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_28 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_30 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_32 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_35 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_38 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_43 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_45 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_47 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_48 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_50 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_53 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_55 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_57 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_58 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_60 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_62 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_63 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_65 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_67 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_68 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_70 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_72 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_73 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_75 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_77 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_78 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_80 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_82 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_83 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_85 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_87 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_88 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_90 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_93 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_95 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_97 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_98 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_100 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_101 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_102 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_103 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_105 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_lift_fresh_where_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                }
            }
        }
    }
}
''')


cpp_fused_view_107 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_108 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_109 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_110 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_lift_fresh_where_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
    }
}
''')


cpp_fused_view_112 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_113 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_115 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_detach_lift_fresh_where_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
    }
}
''')


cpp_fused_view_117 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((128L*x0) + (16384L*(c10::div_floor_integer((x1 + x1_inner), 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(128L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_118 = async_compile.cpp('''
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
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1048576L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = tmp0 * tmp0;
                auto tmp2 = tmp1 * tmp0;
                auto tmp3 = static_cast<float>(0.044715);
                auto tmp4 = at::vec::Vectorized<float>(tmp3);
                auto tmp5 = tmp2 * tmp4;
                auto tmp6 = tmp0 + tmp5;
                auto tmp7 = static_cast<float>(0.7978845608028654);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 * tmp8;
                auto tmp10 = decltype(tmp9)(2) / (decltype(tmp9)(1) + (decltype(tmp9)(-2) * tmp9).exp()) - decltype(tmp9)(1);
                auto tmp11 = static_cast<float>(0.5);
                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                auto tmp13 = tmp0 * tmp12;
                auto tmp14 = static_cast<float>(1.0);
                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                auto tmp16 = tmp10 + tmp15;
                auto tmp17 = tmp13 * tmp16;
                tmp10.store(out_ptr0 + static_cast<long>(x0));
                tmp17.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_120 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (2048L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax__softmax_add_detach_embedding_native_layer_norm_native_layer_norm_backward_nll_loss_forward_121 = async_compile.cpp('''
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
                       float* in_out_ptr40,
                       float* in_out_ptr41,
                       float* in_out_ptr42,
                       float* in_out_ptr43,
                       float* in_out_ptr44,
                       float* in_out_ptr45,
                       float* in_out_ptr46,
                       float* in_out_ptr47,
                       float* in_out_ptr48,
                       float* in_out_ptr49,
                       float* in_out_ptr50,
                       float* in_out_ptr51,
                       float* in_out_ptr52,
                       float* in_out_ptr53,
                       float* in_out_ptr54,
                       float* in_out_ptr55,
                       float* in_out_ptr56,
                       float* in_out_ptr57,
                       float* in_out_ptr58,
                       float* in_out_ptr59,
                       float* in_out_ptr60,
                       float* in_out_ptr61,
                       float* in_out_ptr62,
                       float* in_out_ptr63,
                       float* in_out_ptr64,
                       float* in_out_ptr65,
                       float* in_out_ptr66,
                       float* in_out_ptr67,
                       float* in_out_ptr68,
                       float* in_out_ptr69,
                       float* in_out_ptr70,
                       float* in_out_ptr71,
                       const float* in_ptr0,
                       const long* in_ptr1,
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
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr3,
                       float* out_ptr5)
{
    auto out_ptr4 = in_out_ptr0;
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(127L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50257L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (50257L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(50256L); x1<static_cast<long>(50257L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (50257L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp3 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = std::log(tmp3);
                    auto tmp5 = decltype(tmp2)(tmp2 - tmp4);
                    out_ptr2[static_cast<long>(x1 + (50257L*x0))] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                {
                    long tmp_acc0 = 0;
                    float tmp_acc1 = 0;
                    for(long x0=static_cast<long>(0L); x0<static_cast<long>(127L); x0+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(1L + x0)];
                        auto tmp1 = static_cast<long>(-100);
                        auto tmp2 = tmp0 != tmp1;
                        auto tmp3 = c10::convert<long>(tmp2);
                        auto tmp4 = static_cast<long>(0);
                        auto tmp5 = tmp2 ? tmp0 : tmp4;
                        auto tmp6 = decltype(tmp5)(tmp5 + 50257);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 50257L), "index out of bounds: 0 <= tmp8 < 50257L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (50257L*x0))];
                        auto tmp10 = decltype(tmp9)(-tmp9);
                        auto tmp11 = static_cast<float>(0.0);
                        auto tmp12 = tmp2 ? tmp10 : tmp11;
                        tmp_acc0 = tmp_acc0 + tmp3;
                        tmp_acc1 = tmp_acc1 + tmp12;
                    }
                    out_ptr3[static_cast<long>(0L)] = tmp_acc0;
                    out_ptr4[static_cast<long>(0L)] = tmp_acc1;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = out_ptr3[static_cast<long>(0L)];
                auto tmp2 = out_ptr4[static_cast<long>(0L)];
                auto tmp1 = c10::convert<float>(tmp0);
                auto tmp3 = tmp2 / tmp1;
                out_ptr5[static_cast<long>(0L)] = tmp1;
                in_out_ptr0[static_cast<long>(0L)] = tmp3;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr5 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr7 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr10 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr13 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr16 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr19 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr22 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr25 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr28 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr31 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr32 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr33 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr11[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr34 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr35 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr36 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr37 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr38 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr39 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr13[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr40 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr41 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr42 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr14[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr43 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr44 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr45 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr15[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr46 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr47 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr48 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr16[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr49 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr50 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr51 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr17[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr52 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr53 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr54 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr18[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr55 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr56 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr57 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr19[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr58 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr59 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr60 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr61 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr20[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr61 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr62 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr62 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr63 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr63 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr64 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr21[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr64 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr65 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr66 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr66 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr67 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr22[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr67 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr68 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr68 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr69 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr69 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr70 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr23[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr70 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr71 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(2048.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr71 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343 = args
    args.clear()
    assert_size_stride(primals_1, (50257, 2048), (2048, 1))
    assert_size_stride(primals_2, (2048, 2048), (2048, 1))
    assert_size_stride(primals_3, (2048, ), (1, ))
    assert_size_stride(primals_4, (2048, ), (1, ))
    assert_size_stride(primals_5, (2048, 2048), (2048, 1))
    assert_size_stride(primals_6, (2048, 2048), (2048, 1))
    assert_size_stride(primals_7, (2048, 2048), (2048, 1))
    assert_size_stride(primals_8, (2048, 2048), (2048, 1))
    assert_size_stride(primals_9, (2048, ), (1, ))
    assert_size_stride(primals_10, (2048, ), (1, ))
    assert_size_stride(primals_11, (2048, ), (1, ))
    assert_size_stride(primals_12, (8192, 2048), (2048, 1))
    assert_size_stride(primals_13, (8192, ), (1, ))
    assert_size_stride(primals_14, (2048, 8192), (8192, 1))
    assert_size_stride(primals_15, (2048, ), (1, ))
    assert_size_stride(primals_16, (2048, ), (1, ))
    assert_size_stride(primals_17, (2048, ), (1, ))
    assert_size_stride(primals_18, (2048, 2048), (2048, 1))
    assert_size_stride(primals_19, (2048, 2048), (2048, 1))
    assert_size_stride(primals_20, (2048, 2048), (2048, 1))
    assert_size_stride(primals_21, (2048, 2048), (2048, 1))
    assert_size_stride(primals_22, (2048, ), (1, ))
    assert_size_stride(primals_23, (2048, ), (1, ))
    assert_size_stride(primals_24, (2048, ), (1, ))
    assert_size_stride(primals_25, (8192, 2048), (2048, 1))
    assert_size_stride(primals_26, (8192, ), (1, ))
    assert_size_stride(primals_27, (2048, 8192), (8192, 1))
    assert_size_stride(primals_28, (2048, ), (1, ))
    assert_size_stride(primals_29, (2048, ), (1, ))
    assert_size_stride(primals_30, (2048, ), (1, ))
    assert_size_stride(primals_31, (2048, 2048), (2048, 1))
    assert_size_stride(primals_32, (2048, 2048), (2048, 1))
    assert_size_stride(primals_33, (2048, 2048), (2048, 1))
    assert_size_stride(primals_34, (2048, 2048), (2048, 1))
    assert_size_stride(primals_35, (2048, ), (1, ))
    assert_size_stride(primals_36, (2048, ), (1, ))
    assert_size_stride(primals_37, (2048, ), (1, ))
    assert_size_stride(primals_38, (8192, 2048), (2048, 1))
    assert_size_stride(primals_39, (8192, ), (1, ))
    assert_size_stride(primals_40, (2048, 8192), (8192, 1))
    assert_size_stride(primals_41, (2048, ), (1, ))
    assert_size_stride(primals_42, (2048, ), (1, ))
    assert_size_stride(primals_43, (2048, ), (1, ))
    assert_size_stride(primals_44, (2048, 2048), (2048, 1))
    assert_size_stride(primals_45, (2048, 2048), (2048, 1))
    assert_size_stride(primals_46, (2048, 2048), (2048, 1))
    assert_size_stride(primals_47, (2048, 2048), (2048, 1))
    assert_size_stride(primals_48, (2048, ), (1, ))
    assert_size_stride(primals_49, (2048, ), (1, ))
    assert_size_stride(primals_50, (2048, ), (1, ))
    assert_size_stride(primals_51, (8192, 2048), (2048, 1))
    assert_size_stride(primals_52, (8192, ), (1, ))
    assert_size_stride(primals_53, (2048, 8192), (8192, 1))
    assert_size_stride(primals_54, (2048, ), (1, ))
    assert_size_stride(primals_55, (2048, ), (1, ))
    assert_size_stride(primals_56, (2048, ), (1, ))
    assert_size_stride(primals_57, (2048, 2048), (2048, 1))
    assert_size_stride(primals_58, (2048, 2048), (2048, 1))
    assert_size_stride(primals_59, (2048, 2048), (2048, 1))
    assert_size_stride(primals_60, (2048, 2048), (2048, 1))
    assert_size_stride(primals_61, (2048, ), (1, ))
    assert_size_stride(primals_62, (2048, ), (1, ))
    assert_size_stride(primals_63, (2048, ), (1, ))
    assert_size_stride(primals_64, (8192, 2048), (2048, 1))
    assert_size_stride(primals_65, (8192, ), (1, ))
    assert_size_stride(primals_66, (2048, 8192), (8192, 1))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_68, (2048, ), (1, ))
    assert_size_stride(primals_69, (2048, ), (1, ))
    assert_size_stride(primals_70, (2048, 2048), (2048, 1))
    assert_size_stride(primals_71, (2048, 2048), (2048, 1))
    assert_size_stride(primals_72, (2048, 2048), (2048, 1))
    assert_size_stride(primals_73, (2048, 2048), (2048, 1))
    assert_size_stride(primals_74, (2048, ), (1, ))
    assert_size_stride(primals_75, (2048, ), (1, ))
    assert_size_stride(primals_76, (2048, ), (1, ))
    assert_size_stride(primals_77, (8192, 2048), (2048, 1))
    assert_size_stride(primals_78, (8192, ), (1, ))
    assert_size_stride(primals_79, (2048, 8192), (8192, 1))
    assert_size_stride(primals_80, (2048, ), (1, ))
    assert_size_stride(primals_81, (2048, ), (1, ))
    assert_size_stride(primals_82, (2048, ), (1, ))
    assert_size_stride(primals_83, (2048, 2048), (2048, 1))
    assert_size_stride(primals_84, (2048, 2048), (2048, 1))
    assert_size_stride(primals_85, (2048, 2048), (2048, 1))
    assert_size_stride(primals_86, (2048, 2048), (2048, 1))
    assert_size_stride(primals_87, (2048, ), (1, ))
    assert_size_stride(primals_88, (2048, ), (1, ))
    assert_size_stride(primals_89, (2048, ), (1, ))
    assert_size_stride(primals_90, (8192, 2048), (2048, 1))
    assert_size_stride(primals_91, (8192, ), (1, ))
    assert_size_stride(primals_92, (2048, 8192), (8192, 1))
    assert_size_stride(primals_93, (2048, ), (1, ))
    assert_size_stride(primals_94, (2048, ), (1, ))
    assert_size_stride(primals_95, (2048, ), (1, ))
    assert_size_stride(primals_96, (2048, 2048), (2048, 1))
    assert_size_stride(primals_97, (2048, 2048), (2048, 1))
    assert_size_stride(primals_98, (2048, 2048), (2048, 1))
    assert_size_stride(primals_99, (2048, 2048), (2048, 1))
    assert_size_stride(primals_100, (2048, ), (1, ))
    assert_size_stride(primals_101, (2048, ), (1, ))
    assert_size_stride(primals_102, (2048, ), (1, ))
    assert_size_stride(primals_103, (8192, 2048), (2048, 1))
    assert_size_stride(primals_104, (8192, ), (1, ))
    assert_size_stride(primals_105, (2048, 8192), (8192, 1))
    assert_size_stride(primals_106, (2048, ), (1, ))
    assert_size_stride(primals_107, (2048, ), (1, ))
    assert_size_stride(primals_108, (2048, ), (1, ))
    assert_size_stride(primals_109, (2048, 2048), (2048, 1))
    assert_size_stride(primals_110, (2048, 2048), (2048, 1))
    assert_size_stride(primals_111, (2048, 2048), (2048, 1))
    assert_size_stride(primals_112, (2048, 2048), (2048, 1))
    assert_size_stride(primals_113, (2048, ), (1, ))
    assert_size_stride(primals_114, (2048, ), (1, ))
    assert_size_stride(primals_115, (2048, ), (1, ))
    assert_size_stride(primals_116, (8192, 2048), (2048, 1))
    assert_size_stride(primals_117, (8192, ), (1, ))
    assert_size_stride(primals_118, (2048, 8192), (8192, 1))
    assert_size_stride(primals_119, (2048, ), (1, ))
    assert_size_stride(primals_120, (2048, ), (1, ))
    assert_size_stride(primals_121, (2048, ), (1, ))
    assert_size_stride(primals_122, (2048, 2048), (2048, 1))
    assert_size_stride(primals_123, (2048, 2048), (2048, 1))
    assert_size_stride(primals_124, (2048, 2048), (2048, 1))
    assert_size_stride(primals_125, (2048, 2048), (2048, 1))
    assert_size_stride(primals_126, (2048, ), (1, ))
    assert_size_stride(primals_127, (2048, ), (1, ))
    assert_size_stride(primals_128, (2048, ), (1, ))
    assert_size_stride(primals_129, (8192, 2048), (2048, 1))
    assert_size_stride(primals_130, (8192, ), (1, ))
    assert_size_stride(primals_131, (2048, 8192), (8192, 1))
    assert_size_stride(primals_132, (2048, ), (1, ))
    assert_size_stride(primals_133, (2048, ), (1, ))
    assert_size_stride(primals_134, (2048, ), (1, ))
    assert_size_stride(primals_135, (2048, 2048), (2048, 1))
    assert_size_stride(primals_136, (2048, 2048), (2048, 1))
    assert_size_stride(primals_137, (2048, 2048), (2048, 1))
    assert_size_stride(primals_138, (2048, 2048), (2048, 1))
    assert_size_stride(primals_139, (2048, ), (1, ))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_142, (8192, 2048), (2048, 1))
    assert_size_stride(primals_143, (8192, ), (1, ))
    assert_size_stride(primals_144, (2048, 8192), (8192, 1))
    assert_size_stride(primals_145, (2048, ), (1, ))
    assert_size_stride(primals_146, (2048, ), (1, ))
    assert_size_stride(primals_147, (2048, ), (1, ))
    assert_size_stride(primals_148, (2048, 2048), (2048, 1))
    assert_size_stride(primals_149, (2048, 2048), (2048, 1))
    assert_size_stride(primals_150, (2048, 2048), (2048, 1))
    assert_size_stride(primals_151, (2048, 2048), (2048, 1))
    assert_size_stride(primals_152, (2048, ), (1, ))
    assert_size_stride(primals_153, (2048, ), (1, ))
    assert_size_stride(primals_154, (2048, ), (1, ))
    assert_size_stride(primals_155, (8192, 2048), (2048, 1))
    assert_size_stride(primals_156, (8192, ), (1, ))
    assert_size_stride(primals_157, (2048, 8192), (8192, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_160, (2048, ), (1, ))
    assert_size_stride(primals_161, (2048, 2048), (2048, 1))
    assert_size_stride(primals_162, (2048, 2048), (2048, 1))
    assert_size_stride(primals_163, (2048, 2048), (2048, 1))
    assert_size_stride(primals_164, (2048, 2048), (2048, 1))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_166, (2048, ), (1, ))
    assert_size_stride(primals_167, (2048, ), (1, ))
    assert_size_stride(primals_168, (8192, 2048), (2048, 1))
    assert_size_stride(primals_169, (8192, ), (1, ))
    assert_size_stride(primals_170, (2048, 8192), (8192, 1))
    assert_size_stride(primals_171, (2048, ), (1, ))
    assert_size_stride(primals_172, (2048, ), (1, ))
    assert_size_stride(primals_173, (2048, ), (1, ))
    assert_size_stride(primals_174, (2048, 2048), (2048, 1))
    assert_size_stride(primals_175, (2048, 2048), (2048, 1))
    assert_size_stride(primals_176, (2048, 2048), (2048, 1))
    assert_size_stride(primals_177, (2048, 2048), (2048, 1))
    assert_size_stride(primals_178, (2048, ), (1, ))
    assert_size_stride(primals_179, (2048, ), (1, ))
    assert_size_stride(primals_180, (2048, ), (1, ))
    assert_size_stride(primals_181, (8192, 2048), (2048, 1))
    assert_size_stride(primals_182, (8192, ), (1, ))
    assert_size_stride(primals_183, (2048, 8192), (8192, 1))
    assert_size_stride(primals_184, (2048, ), (1, ))
    assert_size_stride(primals_185, (2048, ), (1, ))
    assert_size_stride(primals_186, (2048, ), (1, ))
    assert_size_stride(primals_187, (2048, 2048), (2048, 1))
    assert_size_stride(primals_188, (2048, 2048), (2048, 1))
    assert_size_stride(primals_189, (2048, 2048), (2048, 1))
    assert_size_stride(primals_190, (2048, 2048), (2048, 1))
    assert_size_stride(primals_191, (2048, ), (1, ))
    assert_size_stride(primals_192, (2048, ), (1, ))
    assert_size_stride(primals_193, (2048, ), (1, ))
    assert_size_stride(primals_194, (8192, 2048), (2048, 1))
    assert_size_stride(primals_195, (8192, ), (1, ))
    assert_size_stride(primals_196, (2048, 8192), (8192, 1))
    assert_size_stride(primals_197, (2048, ), (1, ))
    assert_size_stride(primals_198, (2048, ), (1, ))
    assert_size_stride(primals_199, (2048, ), (1, ))
    assert_size_stride(primals_200, (2048, 2048), (2048, 1))
    assert_size_stride(primals_201, (2048, 2048), (2048, 1))
    assert_size_stride(primals_202, (2048, 2048), (2048, 1))
    assert_size_stride(primals_203, (2048, 2048), (2048, 1))
    assert_size_stride(primals_204, (2048, ), (1, ))
    assert_size_stride(primals_205, (2048, ), (1, ))
    assert_size_stride(primals_206, (2048, ), (1, ))
    assert_size_stride(primals_207, (8192, 2048), (2048, 1))
    assert_size_stride(primals_208, (8192, ), (1, ))
    assert_size_stride(primals_209, (2048, 8192), (8192, 1))
    assert_size_stride(primals_210, (2048, ), (1, ))
    assert_size_stride(primals_211, (2048, ), (1, ))
    assert_size_stride(primals_212, (2048, ), (1, ))
    assert_size_stride(primals_213, (2048, 2048), (2048, 1))
    assert_size_stride(primals_214, (2048, 2048), (2048, 1))
    assert_size_stride(primals_215, (2048, 2048), (2048, 1))
    assert_size_stride(primals_216, (2048, 2048), (2048, 1))
    assert_size_stride(primals_217, (2048, ), (1, ))
    assert_size_stride(primals_218, (2048, ), (1, ))
    assert_size_stride(primals_219, (2048, ), (1, ))
    assert_size_stride(primals_220, (8192, 2048), (2048, 1))
    assert_size_stride(primals_221, (8192, ), (1, ))
    assert_size_stride(primals_222, (2048, 8192), (8192, 1))
    assert_size_stride(primals_223, (2048, ), (1, ))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_225, (2048, ), (1, ))
    assert_size_stride(primals_226, (2048, 2048), (2048, 1))
    assert_size_stride(primals_227, (2048, 2048), (2048, 1))
    assert_size_stride(primals_228, (2048, 2048), (2048, 1))
    assert_size_stride(primals_229, (2048, 2048), (2048, 1))
    assert_size_stride(primals_230, (2048, ), (1, ))
    assert_size_stride(primals_231, (2048, ), (1, ))
    assert_size_stride(primals_232, (2048, ), (1, ))
    assert_size_stride(primals_233, (8192, 2048), (2048, 1))
    assert_size_stride(primals_234, (8192, ), (1, ))
    assert_size_stride(primals_235, (2048, 8192), (8192, 1))
    assert_size_stride(primals_236, (2048, ), (1, ))
    assert_size_stride(primals_237, (2048, ), (1, ))
    assert_size_stride(primals_238, (2048, ), (1, ))
    assert_size_stride(primals_239, (2048, 2048), (2048, 1))
    assert_size_stride(primals_240, (2048, 2048), (2048, 1))
    assert_size_stride(primals_241, (2048, 2048), (2048, 1))
    assert_size_stride(primals_242, (2048, 2048), (2048, 1))
    assert_size_stride(primals_243, (2048, ), (1, ))
    assert_size_stride(primals_244, (2048, ), (1, ))
    assert_size_stride(primals_245, (2048, ), (1, ))
    assert_size_stride(primals_246, (8192, 2048), (2048, 1))
    assert_size_stride(primals_247, (8192, ), (1, ))
    assert_size_stride(primals_248, (2048, 8192), (8192, 1))
    assert_size_stride(primals_249, (2048, ), (1, ))
    assert_size_stride(primals_250, (2048, ), (1, ))
    assert_size_stride(primals_251, (2048, ), (1, ))
    assert_size_stride(primals_252, (2048, 2048), (2048, 1))
    assert_size_stride(primals_253, (2048, 2048), (2048, 1))
    assert_size_stride(primals_254, (2048, 2048), (2048, 1))
    assert_size_stride(primals_255, (2048, 2048), (2048, 1))
    assert_size_stride(primals_256, (2048, ), (1, ))
    assert_size_stride(primals_257, (2048, ), (1, ))
    assert_size_stride(primals_258, (2048, ), (1, ))
    assert_size_stride(primals_259, (8192, 2048), (2048, 1))
    assert_size_stride(primals_260, (8192, ), (1, ))
    assert_size_stride(primals_261, (2048, 8192), (8192, 1))
    assert_size_stride(primals_262, (2048, ), (1, ))
    assert_size_stride(primals_263, (2048, ), (1, ))
    assert_size_stride(primals_264, (2048, ), (1, ))
    assert_size_stride(primals_265, (2048, 2048), (2048, 1))
    assert_size_stride(primals_266, (2048, 2048), (2048, 1))
    assert_size_stride(primals_267, (2048, 2048), (2048, 1))
    assert_size_stride(primals_268, (2048, 2048), (2048, 1))
    assert_size_stride(primals_269, (2048, ), (1, ))
    assert_size_stride(primals_270, (2048, ), (1, ))
    assert_size_stride(primals_271, (2048, ), (1, ))
    assert_size_stride(primals_272, (8192, 2048), (2048, 1))
    assert_size_stride(primals_273, (8192, ), (1, ))
    assert_size_stride(primals_274, (2048, 8192), (8192, 1))
    assert_size_stride(primals_275, (2048, ), (1, ))
    assert_size_stride(primals_276, (2048, ), (1, ))
    assert_size_stride(primals_277, (2048, ), (1, ))
    assert_size_stride(primals_278, (2048, 2048), (2048, 1))
    assert_size_stride(primals_279, (2048, 2048), (2048, 1))
    assert_size_stride(primals_280, (2048, 2048), (2048, 1))
    assert_size_stride(primals_281, (2048, 2048), (2048, 1))
    assert_size_stride(primals_282, (2048, ), (1, ))
    assert_size_stride(primals_283, (2048, ), (1, ))
    assert_size_stride(primals_284, (2048, ), (1, ))
    assert_size_stride(primals_285, (8192, 2048), (2048, 1))
    assert_size_stride(primals_286, (8192, ), (1, ))
    assert_size_stride(primals_287, (2048, 8192), (8192, 1))
    assert_size_stride(primals_288, (2048, ), (1, ))
    assert_size_stride(primals_289, (2048, ), (1, ))
    assert_size_stride(primals_290, (2048, ), (1, ))
    assert_size_stride(primals_291, (2048, 2048), (2048, 1))
    assert_size_stride(primals_292, (2048, 2048), (2048, 1))
    assert_size_stride(primals_293, (2048, 2048), (2048, 1))
    assert_size_stride(primals_294, (2048, 2048), (2048, 1))
    assert_size_stride(primals_295, (2048, ), (1, ))
    assert_size_stride(primals_296, (2048, ), (1, ))
    assert_size_stride(primals_297, (2048, ), (1, ))
    assert_size_stride(primals_298, (8192, 2048), (2048, 1))
    assert_size_stride(primals_299, (8192, ), (1, ))
    assert_size_stride(primals_300, (2048, 8192), (8192, 1))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (2048, ), (1, ))
    assert_size_stride(primals_303, (2048, ), (1, ))
    assert_size_stride(primals_304, (2048, 2048), (2048, 1))
    assert_size_stride(primals_305, (2048, 2048), (2048, 1))
    assert_size_stride(primals_306, (2048, 2048), (2048, 1))
    assert_size_stride(primals_307, (2048, 2048), (2048, 1))
    assert_size_stride(primals_308, (2048, ), (1, ))
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_310, (2048, ), (1, ))
    assert_size_stride(primals_311, (8192, 2048), (2048, 1))
    assert_size_stride(primals_312, (8192, ), (1, ))
    assert_size_stride(primals_313, (2048, 8192), (8192, 1))
    assert_size_stride(primals_314, (2048, ), (1, ))
    assert_size_stride(primals_315, (2048, ), (1, ))
    assert_size_stride(primals_316, (2048, ), (1, ))
    assert_size_stride(primals_317, (50257, 2048), (2048, 1))
    assert_size_stride(primals_318, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_319, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_320, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_321, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_322, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_323, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_324, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_325, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_326, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_327, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_328, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_329, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_330, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_331, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_332, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_333, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_334, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_335, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_336, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_337, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_338, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_339, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_340, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_341, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_342, (1, 128), (128, 1))
    assert_size_stride(primals_343, (1, 128), (128, 1))
    buf0 = empty((1, 128), device='cpu', dtype=torch.int64)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf4 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf5 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_view_0(c_void_p(primals_342.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_4
    buf6 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, reinterpret_tensor(primals_5, (2048, 2048), (1, 2048), 0), out=buf6)
    buf7 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, reinterpret_tensor(primals_6, (2048, 2048), (1, 2048), 0), out=buf7)
    buf8 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, reinterpret_tensor(primals_7, (2048, 2048), (1, 2048), 0), out=buf8)
    buf9 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf6, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf7, (16, 128, 128), (128, 1, 2048), 0), out=buf9)
    buf10 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf11 = reinterpret_tensor(buf9, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf9  # reuse
    buf12 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf13 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_1(c_void_p(buf11.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    buf14 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf8, (16, 128, 128), (128, 2048, 1), 0), out=buf14)
    buf15 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_2(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf14, (128, 2048), (2048, 1), 0); del buf14  # reuse
    # Source Nodes: [attn_output_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_9, buf15, reinterpret_tensor(primals_8, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf16)
    del primals_9
    buf17 = buf1; del buf1  # reuse
    buf18 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf20 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf21 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_view_3(c_void_p(buf16.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del primals_11
    buf22 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_5], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_13, buf21, reinterpret_tensor(primals_12, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf22)
    del primals_13
    buf23 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf24 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_4(c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_15, buf24, reinterpret_tensor(primals_14, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf25)
    del primals_15
    buf26 = reinterpret_tensor(buf25, (1, 128, 2048), (262144, 2048, 1), 0); del buf25  # reuse
    buf27 = buf17; del buf17  # reuse
    buf28 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf30 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf31 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_embedding_native_layer_norm_view_5(c_void_p(buf26.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_1
    del primals_17
    del primals_2
    buf32 = buf16; del buf16  # reuse
    # Source Nodes: [query_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf31, reinterpret_tensor(primals_18, (2048, 2048), (1, 2048), 0), out=buf32)
    buf33 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf31, reinterpret_tensor(primals_19, (2048, 2048), (1, 2048), 0), out=buf33)
    buf34 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_2], Original ATen: [aten.mm]
    extern_kernels.mm(buf31, reinterpret_tensor(primals_20, (2048, 2048), (1, 2048), 0), out=buf34)
    buf35 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf32, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf33, (16, 128, 128), (128, 1, 2048), 0), out=buf35)
    buf36 = buf10; del buf10  # reuse
    buf37 = reinterpret_tensor(buf35, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf35  # reuse
    buf38 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf39 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_6(c_void_p(buf37.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf39, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf34, (16, 128, 128), (128, 2048, 1), 0), out=buf40)
    buf41 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_7(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = reinterpret_tensor(buf40, (128, 2048), (2048, 1), 0); del buf40  # reuse
    # Source Nodes: [attn_output_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_22, buf41, reinterpret_tensor(primals_21, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf42)
    del primals_22
    buf43 = buf27; del buf27  # reuse
    buf44 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf46 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf47 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_8(c_void_p(buf42.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    del primals_24
    buf48 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf47, reinterpret_tensor(primals_25, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf48)
    del primals_26
    buf49 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf50 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_9(c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_28, buf50, reinterpret_tensor(primals_27, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf51)
    del primals_28
    buf52 = buf43; del buf43  # reuse
    buf53 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf55 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf56 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_10(c_void_p(buf42.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    del primals_30
    buf57 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_6], Original ATen: [aten.mm]
    extern_kernels.mm(buf56, reinterpret_tensor(primals_31, (2048, 2048), (1, 2048), 0), out=buf57)
    buf58 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_6], Original ATen: [aten.mm]
    extern_kernels.mm(buf56, reinterpret_tensor(primals_32, (2048, 2048), (1, 2048), 0), out=buf58)
    buf59 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_4], Original ATen: [aten.mm]
    extern_kernels.mm(buf56, reinterpret_tensor(primals_33, (2048, 2048), (1, 2048), 0), out=buf59)
    buf60 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf57, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf58, (16, 128, 128), (128, 1, 2048), 0), out=buf60)
    buf61 = buf36; del buf36  # reuse
    buf62 = reinterpret_tensor(buf60, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf60  # reuse
    buf63 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf64 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_11(c_void_p(buf62.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    buf65 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf64, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf59, (16, 128, 128), (128, 2048, 1), 0), out=buf65)
    buf66 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_12(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = reinterpret_tensor(buf65, (128, 2048), (2048, 1), 0); del buf65  # reuse
    # Source Nodes: [attn_output_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_35, buf66, reinterpret_tensor(primals_34, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf67)
    del primals_35
    buf68 = buf52; del buf52  # reuse
    buf69 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf71 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf72 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_13(c_void_p(buf67.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    del primals_37
    buf73 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_39, buf72, reinterpret_tensor(primals_38, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf73)
    del primals_39
    buf74 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf75 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_14(c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()))
    buf76 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_41, buf75, reinterpret_tensor(primals_40, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf76)
    del primals_41
    buf77 = reinterpret_tensor(buf76, (1, 128, 2048), (262144, 2048, 1), 0); del buf76  # reuse
    buf78 = buf68; del buf68  # reuse
    buf79 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf81 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf82 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_15(c_void_p(buf77.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    del primals_43
    buf83 = buf67; del buf67  # reuse
    # Source Nodes: [query_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf82, reinterpret_tensor(primals_44, (2048, 2048), (1, 2048), 0), out=buf83)
    buf84 = buf51; del buf51  # reuse
    # Source Nodes: [key_9], Original ATen: [aten.mm]
    extern_kernels.mm(buf82, reinterpret_tensor(primals_45, (2048, 2048), (1, 2048), 0), out=buf84)
    buf85 = buf42; del buf42  # reuse
    # Source Nodes: [value_6], Original ATen: [aten.mm]
    extern_kernels.mm(buf82, reinterpret_tensor(primals_46, (2048, 2048), (1, 2048), 0), out=buf85)
    buf86 = reinterpret_tensor(buf26, (16, 128, 128), (16384, 128, 1), 0); del buf26  # reuse
    # Source Nodes: [attn_weights_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf83, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf84, (16, 128, 128), (128, 1, 2048), 0), out=buf86)
    buf87 = buf61; del buf61  # reuse
    buf88 = reinterpret_tensor(buf86, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf86  # reuse
    buf89 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf90 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_16(c_void_p(buf88.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()))
    buf91 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf90, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf85, (16, 128, 128), (128, 2048, 1), 0), out=buf91)
    buf92 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_17(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    buf93 = reinterpret_tensor(buf91, (128, 2048), (2048, 1), 0); del buf91  # reuse
    # Source Nodes: [attn_output_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_48, buf92, reinterpret_tensor(primals_47, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf93)
    del primals_48
    buf94 = buf78; del buf78  # reuse
    buf95 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf97 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf98 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_18(c_void_p(buf93.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    del primals_50
    buf99 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_52, buf98, reinterpret_tensor(primals_51, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf99)
    del primals_52
    buf100 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf101 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_19(c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    buf102 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_34], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf101, reinterpret_tensor(primals_53, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf102)
    del primals_54
    buf103 = buf94; del buf94  # reuse
    buf104 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf106 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf107 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_20(c_void_p(buf93.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del primals_56
    buf108 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_12], Original ATen: [aten.mm]
    extern_kernels.mm(buf107, reinterpret_tensor(primals_57, (2048, 2048), (1, 2048), 0), out=buf108)
    buf109 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_12], Original ATen: [aten.mm]
    extern_kernels.mm(buf107, reinterpret_tensor(primals_58, (2048, 2048), (1, 2048), 0), out=buf109)
    buf110 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_8], Original ATen: [aten.mm]
    extern_kernels.mm(buf107, reinterpret_tensor(primals_59, (2048, 2048), (1, 2048), 0), out=buf110)
    buf111 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf109, (16, 128, 128), (128, 1, 2048), 0), out=buf111)
    buf112 = buf87; del buf87  # reuse
    buf113 = reinterpret_tensor(buf111, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf111  # reuse
    buf114 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf115 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_21(c_void_p(buf113.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf115, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf110, (16, 128, 128), (128, 2048, 1), 0), out=buf116)
    buf117 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_22(c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = reinterpret_tensor(buf116, (128, 2048), (2048, 1), 0); del buf116  # reuse
    # Source Nodes: [attn_output_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_61, buf117, reinterpret_tensor(primals_60, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf118)
    del primals_61
    buf119 = buf103; del buf103  # reuse
    buf120 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf122 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf123 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf118.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del primals_63
    buf124 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_41], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_65, buf123, reinterpret_tensor(primals_64, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf124)
    del primals_65
    buf125 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf126 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_24(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    buf127 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_67, buf126, reinterpret_tensor(primals_66, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf127)
    del primals_67
    buf128 = reinterpret_tensor(buf127, (1, 128, 2048), (262144, 2048, 1), 0); del buf127  # reuse
    buf129 = buf119; del buf119  # reuse
    buf130 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf132 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf133 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf128.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    del primals_69
    buf134 = buf93; del buf93  # reuse
    # Source Nodes: [query_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf133, reinterpret_tensor(primals_70, (2048, 2048), (1, 2048), 0), out=buf134)
    buf135 = reinterpret_tensor(buf77, (128, 2048), (2048, 1), 0); del buf77  # reuse
    # Source Nodes: [key_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf133, reinterpret_tensor(primals_71, (2048, 2048), (1, 2048), 0), out=buf135)
    buf136 = buf118; del buf118  # reuse
    # Source Nodes: [value_10], Original ATen: [aten.mm]
    extern_kernels.mm(buf133, reinterpret_tensor(primals_72, (2048, 2048), (1, 2048), 0), out=buf136)
    buf137 = reinterpret_tensor(buf102, (16, 128, 128), (16384, 128, 1), 0); del buf102  # reuse
    # Source Nodes: [attn_weights_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf134, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf135, (16, 128, 128), (128, 1, 2048), 0), out=buf137)
    buf138 = buf112; del buf112  # reuse
    buf139 = reinterpret_tensor(buf137, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf137  # reuse
    buf140 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf141 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_26(c_void_p(buf139.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf141, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf136, (16, 128, 128), (128, 2048, 1), 0), out=buf142)
    buf143 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_27(c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    buf144 = reinterpret_tensor(buf142, (128, 2048), (2048, 1), 0); del buf142  # reuse
    # Source Nodes: [attn_output_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf143, reinterpret_tensor(primals_73, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf144)
    del primals_74
    buf145 = buf129; del buf129  # reuse
    buf146 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf148 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf149 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_28(c_void_p(buf144.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    del primals_76
    buf150 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_78, buf149, reinterpret_tensor(primals_77, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf150)
    del primals_78
    buf151 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf152 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_29(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    buf153 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_52], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf152, reinterpret_tensor(primals_79, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf153)
    del primals_80
    buf154 = buf145; del buf145  # reuse
    buf155 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf157 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf158 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_30(c_void_p(buf144.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    del primals_82
    buf159 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_18], Original ATen: [aten.mm]
    extern_kernels.mm(buf158, reinterpret_tensor(primals_83, (2048, 2048), (1, 2048), 0), out=buf159)
    buf160 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_18], Original ATen: [aten.mm]
    extern_kernels.mm(buf158, reinterpret_tensor(primals_84, (2048, 2048), (1, 2048), 0), out=buf160)
    buf161 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_12], Original ATen: [aten.mm]
    extern_kernels.mm(buf158, reinterpret_tensor(primals_85, (2048, 2048), (1, 2048), 0), out=buf161)
    buf162 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf159, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf160, (16, 128, 128), (128, 1, 2048), 0), out=buf162)
    buf163 = buf138; del buf138  # reuse
    buf164 = reinterpret_tensor(buf162, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf162  # reuse
    buf165 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf166 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_31(c_void_p(buf164.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf166, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf161, (16, 128, 128), (128, 2048, 1), 0), out=buf167)
    buf168 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_32(c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()))
    buf169 = reinterpret_tensor(buf167, (128, 2048), (2048, 1), 0); del buf167  # reuse
    # Source Nodes: [attn_output_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_87, buf168, reinterpret_tensor(primals_86, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf169)
    del primals_87
    buf170 = buf154; del buf154  # reuse
    buf171 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf173 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf174 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_33(c_void_p(buf169.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del primals_89
    buf175 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_91, buf174, reinterpret_tensor(primals_90, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf175)
    del primals_91
    buf176 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf177 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_34(c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_93, buf177, reinterpret_tensor(primals_92, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf178)
    del primals_93
    buf179 = reinterpret_tensor(buf178, (1, 128, 2048), (262144, 2048, 1), 0); del buf178  # reuse
    buf180 = buf170; del buf170  # reuse
    buf181 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf183 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf184 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf179.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del primals_95
    buf185 = buf169; del buf169  # reuse
    # Source Nodes: [query_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf184, reinterpret_tensor(primals_96, (2048, 2048), (1, 2048), 0), out=buf185)
    buf186 = buf153; del buf153  # reuse
    # Source Nodes: [key_21], Original ATen: [aten.mm]
    extern_kernels.mm(buf184, reinterpret_tensor(primals_97, (2048, 2048), (1, 2048), 0), out=buf186)
    buf187 = buf144; del buf144  # reuse
    # Source Nodes: [value_14], Original ATen: [aten.mm]
    extern_kernels.mm(buf184, reinterpret_tensor(primals_98, (2048, 2048), (1, 2048), 0), out=buf187)
    buf188 = reinterpret_tensor(buf128, (16, 128, 128), (16384, 128, 1), 0); del buf128  # reuse
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf185, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf186, (16, 128, 128), (128, 1, 2048), 0), out=buf188)
    buf189 = buf163; del buf163  # reuse
    buf190 = reinterpret_tensor(buf188, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf188  # reuse
    buf191 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf192 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_36(c_void_p(buf190.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()))
    buf193 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf192, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf187, (16, 128, 128), (128, 2048, 1), 0), out=buf193)
    buf194 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_37(c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    buf195 = reinterpret_tensor(buf193, (128, 2048), (2048, 1), 0); del buf193  # reuse
    # Source Nodes: [attn_output_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf194, reinterpret_tensor(primals_99, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf195)
    del primals_100
    buf196 = buf180; del buf180  # reuse
    buf197 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf199 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf200 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_38(c_void_p(buf195.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    del primals_102
    buf201 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf200, reinterpret_tensor(primals_103, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf201)
    del primals_104
    buf202 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf203 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_39(c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()))
    buf204 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_106, buf203, reinterpret_tensor(primals_105, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf204)
    del primals_106
    buf205 = buf196; del buf196  # reuse
    buf206 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf208 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf209 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_40(c_void_p(buf195.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    del primals_108
    buf210 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_24], Original ATen: [aten.mm]
    extern_kernels.mm(buf209, reinterpret_tensor(primals_109, (2048, 2048), (1, 2048), 0), out=buf210)
    buf211 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_24], Original ATen: [aten.mm]
    extern_kernels.mm(buf209, reinterpret_tensor(primals_110, (2048, 2048), (1, 2048), 0), out=buf211)
    buf212 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_16], Original ATen: [aten.mm]
    extern_kernels.mm(buf209, reinterpret_tensor(primals_111, (2048, 2048), (1, 2048), 0), out=buf212)
    buf213 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf210, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf211, (16, 128, 128), (128, 1, 2048), 0), out=buf213)
    buf214 = buf189; del buf189  # reuse
    buf215 = reinterpret_tensor(buf213, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf213  # reuse
    buf216 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf217 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_41(c_void_p(buf215.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    buf218 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf217, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf212, (16, 128, 128), (128, 2048, 1), 0), out=buf218)
    buf219 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_42(c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = reinterpret_tensor(buf218, (128, 2048), (2048, 1), 0); del buf218  # reuse
    # Source Nodes: [attn_output_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_113, buf219, reinterpret_tensor(primals_112, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf220)
    del primals_113
    buf221 = buf205; del buf205  # reuse
    buf222 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf224 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf225 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_43(c_void_p(buf220.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()))
    del primals_115
    buf226 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_117, buf225, reinterpret_tensor(primals_116, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf226)
    del primals_117
    buf227 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf228 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_44(c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    buf229 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, buf228, reinterpret_tensor(primals_118, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf229)
    del primals_119
    buf230 = reinterpret_tensor(buf229, (1, 128, 2048), (262144, 2048, 1), 0); del buf229  # reuse
    buf231 = buf221; del buf221  # reuse
    buf232 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf234 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf235 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_45(c_void_p(buf230.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del primals_121
    buf236 = buf220; del buf220  # reuse
    # Source Nodes: [query_27], Original ATen: [aten.mm]
    extern_kernels.mm(buf235, reinterpret_tensor(primals_122, (2048, 2048), (1, 2048), 0), out=buf236)
    buf237 = buf204; del buf204  # reuse
    # Source Nodes: [key_27], Original ATen: [aten.mm]
    extern_kernels.mm(buf235, reinterpret_tensor(primals_123, (2048, 2048), (1, 2048), 0), out=buf237)
    buf238 = buf195; del buf195  # reuse
    # Source Nodes: [value_18], Original ATen: [aten.mm]
    extern_kernels.mm(buf235, reinterpret_tensor(primals_124, (2048, 2048), (1, 2048), 0), out=buf238)
    buf239 = reinterpret_tensor(buf179, (16, 128, 128), (16384, 128, 1), 0); del buf179  # reuse
    # Source Nodes: [attn_weights_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf236, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf237, (16, 128, 128), (128, 1, 2048), 0), out=buf239)
    buf240 = buf214; del buf214  # reuse
    buf241 = reinterpret_tensor(buf239, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf239  # reuse
    buf242 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf243 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_46(c_void_p(buf241.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf243, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf238, (16, 128, 128), (128, 2048, 1), 0), out=buf244)
    buf245 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_47(c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf244, (128, 2048), (2048, 1), 0); del buf244  # reuse
    # Source Nodes: [attn_output_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_126, buf245, reinterpret_tensor(primals_125, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf246)
    del primals_126
    buf247 = buf231; del buf231  # reuse
    buf248 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf250 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf251 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_48(c_void_p(buf246.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    del primals_128
    buf252 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf251, reinterpret_tensor(primals_129, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf252)
    del primals_130
    buf253 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf254 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_49(c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    buf255 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf254, reinterpret_tensor(primals_131, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf255)
    del primals_132
    buf256 = buf247; del buf247  # reuse
    buf257 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf259 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf260 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_50(c_void_p(buf246.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del primals_134
    buf261 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_30], Original ATen: [aten.mm]
    extern_kernels.mm(buf260, reinterpret_tensor(primals_135, (2048, 2048), (1, 2048), 0), out=buf261)
    buf262 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_30], Original ATen: [aten.mm]
    extern_kernels.mm(buf260, reinterpret_tensor(primals_136, (2048, 2048), (1, 2048), 0), out=buf262)
    buf263 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf260, reinterpret_tensor(primals_137, (2048, 2048), (1, 2048), 0), out=buf263)
    buf264 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf261, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf262, (16, 128, 128), (128, 1, 2048), 0), out=buf264)
    buf265 = buf240; del buf240  # reuse
    buf266 = reinterpret_tensor(buf264, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf264  # reuse
    buf267 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf268 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_51(c_void_p(buf266.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf268, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf263, (16, 128, 128), (128, 2048, 1), 0), out=buf269)
    buf270 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_52(c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    buf271 = reinterpret_tensor(buf269, (128, 2048), (2048, 1), 0); del buf269  # reuse
    # Source Nodes: [attn_output_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf270, reinterpret_tensor(primals_138, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf271)
    del primals_139
    buf272 = buf256; del buf256  # reuse
    buf273 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf275 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf276 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_53(c_void_p(buf271.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    del primals_141
    buf277 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_143, buf276, reinterpret_tensor(primals_142, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf277)
    del primals_143
    buf278 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf279 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_54(c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    buf280 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_145, buf279, reinterpret_tensor(primals_144, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf280)
    del primals_145
    buf281 = reinterpret_tensor(buf280, (1, 128, 2048), (262144, 2048, 1), 0); del buf280  # reuse
    buf282 = buf272; del buf272  # reuse
    buf283 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf285 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf286 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_55(c_void_p(buf281.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del primals_147
    buf287 = buf271; del buf271  # reuse
    # Source Nodes: [query_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf286, reinterpret_tensor(primals_148, (2048, 2048), (1, 2048), 0), out=buf287)
    buf288 = buf255; del buf255  # reuse
    # Source Nodes: [key_33], Original ATen: [aten.mm]
    extern_kernels.mm(buf286, reinterpret_tensor(primals_149, (2048, 2048), (1, 2048), 0), out=buf288)
    buf289 = buf246; del buf246  # reuse
    # Source Nodes: [value_22], Original ATen: [aten.mm]
    extern_kernels.mm(buf286, reinterpret_tensor(primals_150, (2048, 2048), (1, 2048), 0), out=buf289)
    buf290 = reinterpret_tensor(buf230, (16, 128, 128), (16384, 128, 1), 0); del buf230  # reuse
    # Source Nodes: [attn_weights_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf287, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf288, (16, 128, 128), (128, 1, 2048), 0), out=buf290)
    buf291 = buf265; del buf265  # reuse
    buf292 = reinterpret_tensor(buf290, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf290  # reuse
    buf293 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf294 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_56(c_void_p(buf292.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    buf295 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf294, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf289, (16, 128, 128), (128, 2048, 1), 0), out=buf295)
    buf296 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_57(c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    buf297 = reinterpret_tensor(buf295, (128, 2048), (2048, 1), 0); del buf295  # reuse
    # Source Nodes: [attn_output_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf296, reinterpret_tensor(primals_151, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf297)
    del primals_152
    buf298 = buf282; del buf282  # reuse
    buf299 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf301 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf302 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_58(c_void_p(buf297.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    del primals_154
    buf303 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf302, reinterpret_tensor(primals_155, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf303)
    del primals_156
    buf304 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf305 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_59(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    buf306 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf305, reinterpret_tensor(primals_157, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf306)
    del primals_158
    buf307 = buf298; del buf298  # reuse
    buf308 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf310 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf311 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_60(c_void_p(buf297.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    del primals_160
    buf312 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_36], Original ATen: [aten.mm]
    extern_kernels.mm(buf311, reinterpret_tensor(primals_161, (2048, 2048), (1, 2048), 0), out=buf312)
    buf313 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_36], Original ATen: [aten.mm]
    extern_kernels.mm(buf311, reinterpret_tensor(primals_162, (2048, 2048), (1, 2048), 0), out=buf313)
    buf314 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_24], Original ATen: [aten.mm]
    extern_kernels.mm(buf311, reinterpret_tensor(primals_163, (2048, 2048), (1, 2048), 0), out=buf314)
    buf315 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf312, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf313, (16, 128, 128), (128, 1, 2048), 0), out=buf315)
    buf316 = buf291; del buf291  # reuse
    buf317 = reinterpret_tensor(buf315, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf315  # reuse
    buf318 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf319 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_61(c_void_p(buf317.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    buf320 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf319, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf314, (16, 128, 128), (128, 2048, 1), 0), out=buf320)
    buf321 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_62(c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    buf322 = reinterpret_tensor(buf320, (128, 2048), (2048, 1), 0); del buf320  # reuse
    # Source Nodes: [attn_output_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_165, buf321, reinterpret_tensor(primals_164, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf322)
    del primals_165
    buf323 = buf307; del buf307  # reuse
    buf324 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf326 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf327 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_63(c_void_p(buf322.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    del primals_167
    buf328 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_169, buf327, reinterpret_tensor(primals_168, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf328)
    del primals_169
    buf329 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf330 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_64(c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    buf331 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_171, buf330, reinterpret_tensor(primals_170, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf331)
    del primals_171
    buf332 = reinterpret_tensor(buf331, (1, 128, 2048), (262144, 2048, 1), 0); del buf331  # reuse
    buf333 = buf323; del buf323  # reuse
    buf334 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf336 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf337 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_65(c_void_p(buf332.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    del primals_173
    buf338 = buf322; del buf322  # reuse
    # Source Nodes: [query_39], Original ATen: [aten.mm]
    extern_kernels.mm(buf337, reinterpret_tensor(primals_174, (2048, 2048), (1, 2048), 0), out=buf338)
    buf339 = buf306; del buf306  # reuse
    # Source Nodes: [key_39], Original ATen: [aten.mm]
    extern_kernels.mm(buf337, reinterpret_tensor(primals_175, (2048, 2048), (1, 2048), 0), out=buf339)
    buf340 = buf297; del buf297  # reuse
    # Source Nodes: [value_26], Original ATen: [aten.mm]
    extern_kernels.mm(buf337, reinterpret_tensor(primals_176, (2048, 2048), (1, 2048), 0), out=buf340)
    buf341 = reinterpret_tensor(buf281, (16, 128, 128), (16384, 128, 1), 0); del buf281  # reuse
    # Source Nodes: [attn_weights_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf338, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf339, (16, 128, 128), (128, 1, 2048), 0), out=buf341)
    buf342 = buf316; del buf316  # reuse
    buf343 = reinterpret_tensor(buf341, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf341  # reuse
    buf344 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf345 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_66(c_void_p(buf343.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()))
    buf346 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf345, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf340, (16, 128, 128), (128, 2048, 1), 0), out=buf346)
    buf347 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_67(c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()))
    buf348 = reinterpret_tensor(buf346, (128, 2048), (2048, 1), 0); del buf346  # reuse
    # Source Nodes: [attn_output_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, buf347, reinterpret_tensor(primals_177, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf348)
    del primals_178
    buf349 = buf333; del buf333  # reuse
    buf350 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf352 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf353 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_68(c_void_p(buf348.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_180.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    del primals_180
    buf354 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf353, reinterpret_tensor(primals_181, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf354)
    del primals_182
    buf355 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf356 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_69(c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()))
    buf357 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_184, buf356, reinterpret_tensor(primals_183, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf357)
    del primals_184
    buf358 = buf349; del buf349  # reuse
    buf359 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf361 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf362 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_70(c_void_p(buf348.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_186.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    del primals_186
    buf363 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_42], Original ATen: [aten.mm]
    extern_kernels.mm(buf362, reinterpret_tensor(primals_187, (2048, 2048), (1, 2048), 0), out=buf363)
    buf364 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_42], Original ATen: [aten.mm]
    extern_kernels.mm(buf362, reinterpret_tensor(primals_188, (2048, 2048), (1, 2048), 0), out=buf364)
    buf365 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_28], Original ATen: [aten.mm]
    extern_kernels.mm(buf362, reinterpret_tensor(primals_189, (2048, 2048), (1, 2048), 0), out=buf365)
    buf366 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf363, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf364, (16, 128, 128), (128, 1, 2048), 0), out=buf366)
    buf367 = buf342; del buf342  # reuse
    buf368 = reinterpret_tensor(buf366, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf366  # reuse
    buf369 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf370 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_71(c_void_p(buf368.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    buf371 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf370, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf365, (16, 128, 128), (128, 2048, 1), 0), out=buf371)
    buf372 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_72(c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    buf373 = reinterpret_tensor(buf371, (128, 2048), (2048, 1), 0); del buf371  # reuse
    # Source Nodes: [attn_output_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_191, buf372, reinterpret_tensor(primals_190, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf373)
    del primals_191
    buf374 = buf358; del buf358  # reuse
    buf375 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf377 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf378 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_73(c_void_p(buf373.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    del primals_193
    buf379 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_195, buf378, reinterpret_tensor(primals_194, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf379)
    del primals_195
    buf380 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf381 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_74(c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_197, buf381, reinterpret_tensor(primals_196, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf382)
    del primals_197
    buf383 = reinterpret_tensor(buf382, (1, 128, 2048), (262144, 2048, 1), 0); del buf382  # reuse
    buf384 = buf374; del buf374  # reuse
    buf385 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf387 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf388 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_75(c_void_p(buf383.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_198.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()))
    del primals_199
    buf389 = buf373; del buf373  # reuse
    # Source Nodes: [query_45], Original ATen: [aten.mm]
    extern_kernels.mm(buf388, reinterpret_tensor(primals_200, (2048, 2048), (1, 2048), 0), out=buf389)
    buf390 = buf357; del buf357  # reuse
    # Source Nodes: [key_45], Original ATen: [aten.mm]
    extern_kernels.mm(buf388, reinterpret_tensor(primals_201, (2048, 2048), (1, 2048), 0), out=buf390)
    buf391 = buf348; del buf348  # reuse
    # Source Nodes: [value_30], Original ATen: [aten.mm]
    extern_kernels.mm(buf388, reinterpret_tensor(primals_202, (2048, 2048), (1, 2048), 0), out=buf391)
    buf392 = reinterpret_tensor(buf332, (16, 128, 128), (16384, 128, 1), 0); del buf332  # reuse
    # Source Nodes: [attn_weights_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf389, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf390, (16, 128, 128), (128, 1, 2048), 0), out=buf392)
    buf393 = buf367; del buf367  # reuse
    buf394 = reinterpret_tensor(buf392, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf392  # reuse
    buf395 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf396 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_76(c_void_p(buf394.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    buf397 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf396, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf391, (16, 128, 128), (128, 2048, 1), 0), out=buf397)
    buf398 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_77(c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()))
    buf399 = reinterpret_tensor(buf397, (128, 2048), (2048, 1), 0); del buf397  # reuse
    # Source Nodes: [attn_output_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_204, buf398, reinterpret_tensor(primals_203, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf399)
    del primals_204
    buf400 = buf384; del buf384  # reuse
    buf401 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf403 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf404 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_78(c_void_p(buf399.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    del primals_206
    buf405 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_208, buf404, reinterpret_tensor(primals_207, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf405)
    del primals_208
    buf406 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf407 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_79(c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_210, buf407, reinterpret_tensor(primals_209, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf408)
    del primals_210
    buf409 = buf400; del buf400  # reuse
    buf410 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf412 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf413 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_80(c_void_p(buf399.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()))
    del primals_212
    buf414 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_48], Original ATen: [aten.mm]
    extern_kernels.mm(buf413, reinterpret_tensor(primals_213, (2048, 2048), (1, 2048), 0), out=buf414)
    buf415 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_48], Original ATen: [aten.mm]
    extern_kernels.mm(buf413, reinterpret_tensor(primals_214, (2048, 2048), (1, 2048), 0), out=buf415)
    buf416 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_32], Original ATen: [aten.mm]
    extern_kernels.mm(buf413, reinterpret_tensor(primals_215, (2048, 2048), (1, 2048), 0), out=buf416)
    buf417 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_96], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf414, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf415, (16, 128, 128), (128, 1, 2048), 0), out=buf417)
    buf418 = buf393; del buf393  # reuse
    buf419 = reinterpret_tensor(buf417, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf417  # reuse
    buf420 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf421 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_81(c_void_p(buf419.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    buf422 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_96], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf421, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf416, (16, 128, 128), (128, 2048, 1), 0), out=buf422)
    buf423 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_82(c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()))
    buf424 = reinterpret_tensor(buf422, (128, 2048), (2048, 1), 0); del buf422  # reuse
    # Source Nodes: [attn_output_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_217, buf423, reinterpret_tensor(primals_216, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf424)
    del primals_217
    buf425 = buf409; del buf409  # reuse
    buf426 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf428 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf429 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_83(c_void_p(buf424.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    del primals_219
    buf430 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_221, buf429, reinterpret_tensor(primals_220, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf430)
    del primals_221
    buf431 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf432 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_84(c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()))
    buf433 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_223, buf432, reinterpret_tensor(primals_222, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf433)
    del primals_223
    buf434 = reinterpret_tensor(buf433, (1, 128, 2048), (262144, 2048, 1), 0); del buf433  # reuse
    buf435 = buf425; del buf425  # reuse
    buf436 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf438 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf439 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_85(c_void_p(buf434.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()))
    del primals_225
    buf440 = buf424; del buf424  # reuse
    # Source Nodes: [query_51], Original ATen: [aten.mm]
    extern_kernels.mm(buf439, reinterpret_tensor(primals_226, (2048, 2048), (1, 2048), 0), out=buf440)
    buf441 = buf408; del buf408  # reuse
    # Source Nodes: [key_51], Original ATen: [aten.mm]
    extern_kernels.mm(buf439, reinterpret_tensor(primals_227, (2048, 2048), (1, 2048), 0), out=buf441)
    buf442 = buf399; del buf399  # reuse
    # Source Nodes: [value_34], Original ATen: [aten.mm]
    extern_kernels.mm(buf439, reinterpret_tensor(primals_228, (2048, 2048), (1, 2048), 0), out=buf442)
    buf443 = reinterpret_tensor(buf383, (16, 128, 128), (16384, 128, 1), 0); del buf383  # reuse
    # Source Nodes: [attn_weights_102], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf440, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf441, (16, 128, 128), (128, 1, 2048), 0), out=buf443)
    buf444 = buf418; del buf418  # reuse
    buf445 = reinterpret_tensor(buf443, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf443  # reuse
    buf446 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf447 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_86(c_void_p(buf445.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()))
    buf448 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_102], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf447, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf442, (16, 128, 128), (128, 2048, 1), 0), out=buf448)
    buf449 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_87(c_void_p(buf448.data_ptr()), c_void_p(buf449.data_ptr()))
    buf450 = reinterpret_tensor(buf448, (128, 2048), (2048, 1), 0); del buf448  # reuse
    # Source Nodes: [attn_output_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_230, buf449, reinterpret_tensor(primals_229, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf450)
    del primals_230
    buf451 = buf435; del buf435  # reuse
    buf452 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf454 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf455 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_88(c_void_p(buf450.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()))
    del primals_232
    buf456 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_158], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_234, buf455, reinterpret_tensor(primals_233, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf456)
    del primals_234
    buf457 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf458 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_89(c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    buf459 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_160], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_236, buf458, reinterpret_tensor(primals_235, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf459)
    del primals_236
    buf460 = buf451; del buf451  # reuse
    buf461 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf463 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf464 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_90(c_void_p(buf450.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()))
    del primals_238
    buf465 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_54], Original ATen: [aten.mm]
    extern_kernels.mm(buf464, reinterpret_tensor(primals_239, (2048, 2048), (1, 2048), 0), out=buf465)
    buf466 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_54], Original ATen: [aten.mm]
    extern_kernels.mm(buf464, reinterpret_tensor(primals_240, (2048, 2048), (1, 2048), 0), out=buf466)
    buf467 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_36], Original ATen: [aten.mm]
    extern_kernels.mm(buf464, reinterpret_tensor(primals_241, (2048, 2048), (1, 2048), 0), out=buf467)
    buf468 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_108], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf465, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf466, (16, 128, 128), (128, 1, 2048), 0), out=buf468)
    buf469 = buf444; del buf444  # reuse
    buf470 = reinterpret_tensor(buf468, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf468  # reuse
    buf471 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf472 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_91(c_void_p(buf470.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()))
    buf473 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_108], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf472, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf467, (16, 128, 128), (128, 2048, 1), 0), out=buf473)
    buf474 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_92(c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()))
    buf475 = reinterpret_tensor(buf473, (128, 2048), (2048, 1), 0); del buf473  # reuse
    # Source Nodes: [attn_output_110], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_243, buf474, reinterpret_tensor(primals_242, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf475)
    del primals_243
    buf476 = buf460; del buf460  # reuse
    buf477 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf479 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf480 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_93(c_void_p(buf475.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()))
    del primals_245
    buf481 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_167], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_247, buf480, reinterpret_tensor(primals_246, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf481)
    del primals_247
    buf482 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf483 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_94(c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()))
    buf484 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_169], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_249, buf483, reinterpret_tensor(primals_248, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf484)
    del primals_249
    buf485 = reinterpret_tensor(buf484, (1, 128, 2048), (262144, 2048, 1), 0); del buf484  # reuse
    buf486 = buf476; del buf476  # reuse
    buf487 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf489 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf490 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_95(c_void_p(buf485.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()))
    del primals_251
    buf491 = buf475; del buf475  # reuse
    # Source Nodes: [query_57], Original ATen: [aten.mm]
    extern_kernels.mm(buf490, reinterpret_tensor(primals_252, (2048, 2048), (1, 2048), 0), out=buf491)
    buf492 = buf459; del buf459  # reuse
    # Source Nodes: [key_57], Original ATen: [aten.mm]
    extern_kernels.mm(buf490, reinterpret_tensor(primals_253, (2048, 2048), (1, 2048), 0), out=buf492)
    buf493 = buf450; del buf450  # reuse
    # Source Nodes: [value_38], Original ATen: [aten.mm]
    extern_kernels.mm(buf490, reinterpret_tensor(primals_254, (2048, 2048), (1, 2048), 0), out=buf493)
    buf494 = reinterpret_tensor(buf434, (16, 128, 128), (16384, 128, 1), 0); del buf434  # reuse
    # Source Nodes: [attn_weights_114], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf491, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf492, (16, 128, 128), (128, 1, 2048), 0), out=buf494)
    buf495 = buf469; del buf469  # reuse
    buf496 = reinterpret_tensor(buf494, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf494  # reuse
    buf497 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf498 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_96(c_void_p(buf496.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()))
    buf499 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_114], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf498, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf493, (16, 128, 128), (128, 2048, 1), 0), out=buf499)
    buf500 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_97(c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    buf501 = reinterpret_tensor(buf499, (128, 2048), (2048, 1), 0); del buf499  # reuse
    # Source Nodes: [attn_output_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_256, buf500, reinterpret_tensor(primals_255, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf501)
    del primals_256
    buf502 = buf486; del buf486  # reuse
    buf503 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf505 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf506 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_98(c_void_p(buf501.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()))
    del primals_258
    buf507 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_176], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_260, buf506, reinterpret_tensor(primals_259, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf507)
    del primals_260
    buf508 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf509 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_99(c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()))
    buf510 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_178], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_262, buf509, reinterpret_tensor(primals_261, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf510)
    del primals_262
    buf511 = buf502; del buf502  # reuse
    buf512 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf514 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf515 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_100(c_void_p(buf501.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()))
    del primals_264
    buf516 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_60], Original ATen: [aten.mm]
    extern_kernels.mm(buf515, reinterpret_tensor(primals_265, (2048, 2048), (1, 2048), 0), out=buf516)
    buf517 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_60], Original ATen: [aten.mm]
    extern_kernels.mm(buf515, reinterpret_tensor(primals_266, (2048, 2048), (1, 2048), 0), out=buf517)
    buf518 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_40], Original ATen: [aten.mm]
    extern_kernels.mm(buf515, reinterpret_tensor(primals_267, (2048, 2048), (1, 2048), 0), out=buf518)
    buf519 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf516, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf517, (16, 128, 128), (128, 1, 2048), 0), out=buf519)
    buf520 = buf495; del buf495  # reuse
    buf521 = reinterpret_tensor(buf519, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf519  # reuse
    buf522 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf523 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_101(c_void_p(buf521.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()))
    buf524 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf523, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf518, (16, 128, 128), (128, 2048, 1), 0), out=buf524)
    buf525 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_102(c_void_p(buf524.data_ptr()), c_void_p(buf525.data_ptr()))
    buf526 = reinterpret_tensor(buf524, (128, 2048), (2048, 1), 0); del buf524  # reuse
    # Source Nodes: [attn_output_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_269, buf525, reinterpret_tensor(primals_268, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf526)
    del primals_269
    buf527 = buf511; del buf511  # reuse
    buf528 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf530 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf531 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_103(c_void_p(buf526.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf531.data_ptr()))
    del primals_271
    buf532 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_185], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_273, buf531, reinterpret_tensor(primals_272, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf532)
    del primals_273
    buf533 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf534 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_104(c_void_p(buf532.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()))
    buf535 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_187], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_275, buf534, reinterpret_tensor(primals_274, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf535)
    del primals_275
    buf536 = reinterpret_tensor(buf535, (1, 128, 2048), (262144, 2048, 1), 0); del buf535  # reuse
    buf537 = buf527; del buf527  # reuse
    buf538 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf540 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf541 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_105(c_void_p(buf536.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf541.data_ptr()))
    del primals_277
    buf542 = buf526; del buf526  # reuse
    # Source Nodes: [query_63], Original ATen: [aten.mm]
    extern_kernels.mm(buf541, reinterpret_tensor(primals_278, (2048, 2048), (1, 2048), 0), out=buf542)
    buf543 = buf510; del buf510  # reuse
    # Source Nodes: [key_63], Original ATen: [aten.mm]
    extern_kernels.mm(buf541, reinterpret_tensor(primals_279, (2048, 2048), (1, 2048), 0), out=buf543)
    buf544 = buf501; del buf501  # reuse
    # Source Nodes: [value_42], Original ATen: [aten.mm]
    extern_kernels.mm(buf541, reinterpret_tensor(primals_280, (2048, 2048), (1, 2048), 0), out=buf544)
    buf545 = reinterpret_tensor(buf485, (16, 128, 128), (16384, 128, 1), 0); del buf485  # reuse
    # Source Nodes: [attn_weights_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf542, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf543, (16, 128, 128), (128, 1, 2048), 0), out=buf545)
    buf546 = buf520; del buf520  # reuse
    buf547 = reinterpret_tensor(buf545, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf545  # reuse
    buf548 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf549 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_lift_fresh_where_106(c_void_p(buf547.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()))
    buf550 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_output_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf549, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf544, (16, 128, 128), (128, 2048, 1), 0), out=buf550)
    buf551 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_107(c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()))
    buf552 = reinterpret_tensor(buf550, (128, 2048), (2048, 1), 0); del buf550  # reuse
    # Source Nodes: [attn_output_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_282, buf551, reinterpret_tensor(primals_281, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf552)
    del primals_282
    buf553 = buf537; del buf537  # reuse
    buf554 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf556 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf557 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_108(c_void_p(buf552.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()))
    del primals_284
    buf558 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_194], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_286, buf557, reinterpret_tensor(primals_285, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf558)
    del primals_286
    buf559 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf560 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_109(c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()))
    buf561 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_196], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_288, buf560, reinterpret_tensor(primals_287, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf561)
    del primals_288
    buf562 = buf553; del buf553  # reuse
    buf563 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf565 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf566 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_110(c_void_p(buf552.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf566.data_ptr()))
    del primals_290
    buf567 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_66], Original ATen: [aten.mm]
    extern_kernels.mm(buf566, reinterpret_tensor(primals_291, (2048, 2048), (1, 2048), 0), out=buf567)
    buf568 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_66], Original ATen: [aten.mm]
    extern_kernels.mm(buf566, reinterpret_tensor(primals_292, (2048, 2048), (1, 2048), 0), out=buf568)
    buf569 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_44], Original ATen: [aten.mm]
    extern_kernels.mm(buf566, reinterpret_tensor(primals_293, (2048, 2048), (1, 2048), 0), out=buf569)
    buf570 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_132], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf567, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf568, (16, 128, 128), (128, 1, 2048), 0), out=buf570)
    buf571 = buf546; del buf546  # reuse
    buf572 = reinterpret_tensor(buf570, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf570  # reuse
    buf573 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf574 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    buf630 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_detach_lift_fresh_where_111(c_void_p(buf572.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf630.data_ptr()))
    buf575 = reinterpret_tensor(buf572, (16, 128, 128), (16384, 128, 1), 0); del buf572  # reuse
    # Source Nodes: [attn_output_132], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf574, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf569, (16, 128, 128), (128, 2048, 1), 0), out=buf575)
    buf576 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_112(c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()))
    buf577 = reinterpret_tensor(buf575, (128, 2048), (2048, 1), 0); del buf575  # reuse
    # Source Nodes: [attn_output_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_295, buf576, reinterpret_tensor(primals_294, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf577)
    del primals_295
    buf578 = buf562; del buf562  # reuse
    buf579 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf581 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf582 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_113(c_void_p(buf577.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()))
    del primals_297
    buf583 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_203], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_299, buf582, reinterpret_tensor(primals_298, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf583)
    del primals_299
    buf584 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf585 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_114(c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()))
    buf586 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_205], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_301, buf585, reinterpret_tensor(primals_300, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf586)
    del primals_301
    buf587 = reinterpret_tensor(buf586, (1, 128, 2048), (262144, 2048, 1), 0); del buf586  # reuse
    buf588 = buf578; del buf578  # reuse
    buf589 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf591 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf592 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_115(c_void_p(buf587.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf536.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()))
    del primals_303
    buf593 = buf577; del buf577  # reuse
    # Source Nodes: [query_69], Original ATen: [aten.mm]
    extern_kernels.mm(buf592, reinterpret_tensor(primals_304, (2048, 2048), (1, 2048), 0), out=buf593)
    buf594 = buf561; del buf561  # reuse
    # Source Nodes: [key_69], Original ATen: [aten.mm]
    extern_kernels.mm(buf592, reinterpret_tensor(primals_305, (2048, 2048), (1, 2048), 0), out=buf594)
    buf595 = buf552; del buf552  # reuse
    # Source Nodes: [value_46], Original ATen: [aten.mm]
    extern_kernels.mm(buf592, reinterpret_tensor(primals_306, (2048, 2048), (1, 2048), 0), out=buf595)
    buf596 = reinterpret_tensor(buf536, (16, 128, 128), (16384, 128, 1), 0); del buf536  # reuse
    # Source Nodes: [attn_weights_138], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf593, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(buf594, (16, 128, 128), (128, 1, 2048), 0), out=buf596)
    buf597 = buf573; del buf573  # reuse
    buf598 = reinterpret_tensor(buf596, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf596  # reuse
    buf599 = buf571; del buf571  # reuse
    buf600 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    buf627 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_detach_lift_fresh_where_116(c_void_p(buf598.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf627.data_ptr()))
    del buf597
    del buf599
    buf601 = reinterpret_tensor(buf598, (16, 128, 128), (16384, 128, 1), 0); del buf598  # reuse
    # Source Nodes: [attn_output_138], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf600, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf595, (16, 128, 128), (128, 2048, 1), 0), out=buf601)
    buf602 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_view_117(c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()))
    buf603 = reinterpret_tensor(buf601, (128, 2048), (2048, 1), 0); del buf601  # reuse
    # Source Nodes: [attn_output_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_308, buf602, reinterpret_tensor(primals_307, (2048, 2048), (1, 2048), 0), alpha=1, beta=1, out=buf603)
    del primals_308
    buf604 = buf588; del buf588  # reuse
    buf605 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf607 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf608 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_118(c_void_p(buf603.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf608.data_ptr()))
    del primals_310
    buf609 = empty((128, 8192), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_212], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_312, buf608, reinterpret_tensor(primals_311, (2048, 8192), (1, 2048), 0), alpha=1, beta=1, out=buf609)
    del primals_312
    buf610 = empty((1, 128, 8192), device='cpu', dtype=torch.float32)
    buf611 = empty((128, 8192), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_119(c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()))
    buf612 = empty((128, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_214], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_314, buf611, reinterpret_tensor(primals_313, (8192, 2048), (1, 8192), 0), alpha=1, beta=1, out=buf612)
    del primals_314
    buf613 = buf604; del buf604  # reuse
    buf614 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf616 = empty((1, 128, 2048), device='cpu', dtype=torch.float32)
    buf617 = empty((128, 2048), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_120(c_void_p(buf603.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()))
    del buf587
    del buf603
    del buf612
    del buf613
    del primals_316
    buf618 = empty((128, 50257), device='cpu', dtype=torch.float32)
    # Source Nodes: [lm_logits], Original ATen: [aten.mm]
    extern_kernels.mm(buf617, reinterpret_tensor(primals_317, (2048, 50257), (1, 2048), 0), out=buf618)
    buf619 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.float32)
    buf620 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.float32)
    buf621 = empty((127, 50257), device='cpu', dtype=torch.float32)
    buf622 = empty((), device='cpu', dtype=torch.int64)
    buf624 = empty((), device='cpu', dtype=torch.float32)
    buf623 = empty((), device='cpu', dtype=torch.float32)
    buf698 = buf624; del buf624  # reuse
    buf625 = reinterpret_tensor(buf614, (1, 128, 1), (128, 1, 1), 0); del buf614  # reuse
    buf626 = reinterpret_tensor(buf605, (1, 128, 1), (128, 1, 1), 0); del buf605  # reuse
    buf628 = reinterpret_tensor(buf589, (1, 128, 1), (128, 1, 1), 0); del buf589  # reuse
    buf629 = reinterpret_tensor(buf579, (1, 128, 1), (128, 1, 1), 0); del buf579  # reuse
    buf631 = reinterpret_tensor(buf563, (1, 128, 1), (128, 1, 1), 0); del buf563  # reuse
    buf632 = reinterpret_tensor(buf554, (1, 128, 1), (128, 1, 1), 0); del buf554  # reuse
    buf633 = buf547; del buf547  # reuse
    buf634 = reinterpret_tensor(buf538, (1, 128, 1), (128, 1, 1), 0); del buf538  # reuse
    buf635 = reinterpret_tensor(buf528, (1, 128, 1), (128, 1, 1), 0); del buf528  # reuse
    buf636 = buf521; del buf521  # reuse
    buf637 = reinterpret_tensor(buf512, (1, 128, 1), (128, 1, 1), 0); del buf512  # reuse
    buf638 = reinterpret_tensor(buf503, (1, 128, 1), (128, 1, 1), 0); del buf503  # reuse
    buf639 = buf496; del buf496  # reuse
    buf640 = reinterpret_tensor(buf487, (1, 128, 1), (128, 1, 1), 0); del buf487  # reuse
    buf641 = reinterpret_tensor(buf477, (1, 128, 1), (128, 1, 1), 0); del buf477  # reuse
    buf642 = buf470; del buf470  # reuse
    buf643 = reinterpret_tensor(buf461, (1, 128, 1), (128, 1, 1), 0); del buf461  # reuse
    buf644 = reinterpret_tensor(buf452, (1, 128, 1), (128, 1, 1), 0); del buf452  # reuse
    buf645 = buf445; del buf445  # reuse
    buf646 = reinterpret_tensor(buf436, (1, 128, 1), (128, 1, 1), 0); del buf436  # reuse
    buf647 = reinterpret_tensor(buf426, (1, 128, 1), (128, 1, 1), 0); del buf426  # reuse
    buf648 = buf419; del buf419  # reuse
    buf649 = reinterpret_tensor(buf410, (1, 128, 1), (128, 1, 1), 0); del buf410  # reuse
    buf650 = reinterpret_tensor(buf401, (1, 128, 1), (128, 1, 1), 0); del buf401  # reuse
    buf651 = buf394; del buf394  # reuse
    buf652 = reinterpret_tensor(buf385, (1, 128, 1), (128, 1, 1), 0); del buf385  # reuse
    buf653 = reinterpret_tensor(buf375, (1, 128, 1), (128, 1, 1), 0); del buf375  # reuse
    buf654 = buf368; del buf368  # reuse
    buf655 = reinterpret_tensor(buf359, (1, 128, 1), (128, 1, 1), 0); del buf359  # reuse
    buf656 = reinterpret_tensor(buf350, (1, 128, 1), (128, 1, 1), 0); del buf350  # reuse
    buf657 = buf343; del buf343  # reuse
    buf658 = reinterpret_tensor(buf334, (1, 128, 1), (128, 1, 1), 0); del buf334  # reuse
    buf659 = reinterpret_tensor(buf324, (1, 128, 1), (128, 1, 1), 0); del buf324  # reuse
    buf660 = buf317; del buf317  # reuse
    buf661 = reinterpret_tensor(buf308, (1, 128, 1), (128, 1, 1), 0); del buf308  # reuse
    buf662 = reinterpret_tensor(buf299, (1, 128, 1), (128, 1, 1), 0); del buf299  # reuse
    buf663 = buf292; del buf292  # reuse
    buf664 = reinterpret_tensor(buf283, (1, 128, 1), (128, 1, 1), 0); del buf283  # reuse
    buf665 = reinterpret_tensor(buf273, (1, 128, 1), (128, 1, 1), 0); del buf273  # reuse
    buf666 = buf266; del buf266  # reuse
    buf667 = reinterpret_tensor(buf257, (1, 128, 1), (128, 1, 1), 0); del buf257  # reuse
    buf668 = reinterpret_tensor(buf248, (1, 128, 1), (128, 1, 1), 0); del buf248  # reuse
    buf669 = buf241; del buf241  # reuse
    buf670 = reinterpret_tensor(buf232, (1, 128, 1), (128, 1, 1), 0); del buf232  # reuse
    buf671 = reinterpret_tensor(buf222, (1, 128, 1), (128, 1, 1), 0); del buf222  # reuse
    buf672 = buf215; del buf215  # reuse
    buf673 = reinterpret_tensor(buf206, (1, 128, 1), (128, 1, 1), 0); del buf206  # reuse
    buf674 = reinterpret_tensor(buf197, (1, 128, 1), (128, 1, 1), 0); del buf197  # reuse
    buf675 = buf190; del buf190  # reuse
    buf676 = reinterpret_tensor(buf181, (1, 128, 1), (128, 1, 1), 0); del buf181  # reuse
    buf677 = reinterpret_tensor(buf171, (1, 128, 1), (128, 1, 1), 0); del buf171  # reuse
    buf678 = buf164; del buf164  # reuse
    buf679 = reinterpret_tensor(buf155, (1, 128, 1), (128, 1, 1), 0); del buf155  # reuse
    buf680 = reinterpret_tensor(buf146, (1, 128, 1), (128, 1, 1), 0); del buf146  # reuse
    buf681 = buf139; del buf139  # reuse
    buf682 = reinterpret_tensor(buf130, (1, 128, 1), (128, 1, 1), 0); del buf130  # reuse
    buf683 = reinterpret_tensor(buf120, (1, 128, 1), (128, 1, 1), 0); del buf120  # reuse
    buf684 = buf113; del buf113  # reuse
    buf685 = reinterpret_tensor(buf104, (1, 128, 1), (128, 1, 1), 0); del buf104  # reuse
    buf686 = reinterpret_tensor(buf95, (1, 128, 1), (128, 1, 1), 0); del buf95  # reuse
    buf687 = buf88; del buf88  # reuse
    buf688 = reinterpret_tensor(buf79, (1, 128, 1), (128, 1, 1), 0); del buf79  # reuse
    buf689 = reinterpret_tensor(buf69, (1, 128, 1), (128, 1, 1), 0); del buf69  # reuse
    buf690 = buf62; del buf62  # reuse
    buf691 = reinterpret_tensor(buf53, (1, 128, 1), (128, 1, 1), 0); del buf53  # reuse
    buf692 = reinterpret_tensor(buf44, (1, 128, 1), (128, 1, 1), 0); del buf44  # reuse
    buf693 = buf37; del buf37  # reuse
    buf694 = reinterpret_tensor(buf28, (1, 128, 1), (128, 1, 1), 0); del buf28  # reuse
    buf695 = reinterpret_tensor(buf18, (1, 128, 1), (128, 1, 1), 0); del buf18  # reuse
    buf696 = buf11; del buf11  # reuse
    buf697 = reinterpret_tensor(buf2, (1, 128, 1), (128, 1, 1), 0); del buf2  # reuse
    cpp_fused__log_softmax__softmax_add_detach_embedding_native_layer_norm_native_layer_norm_backward_nll_loss_forward_121(c_void_p(buf698.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf633.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf644.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf692.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()))
    return (buf698, reinterpret_tensor(buf618, (1, 128, 50257), (6432896, 50257, 1), 0), reinterpret_tensor(buf7, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf8, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf33, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf34, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf58, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf59, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf84, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf85, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf109, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf110, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf135, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf136, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf160, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf161, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf186, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf187, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf211, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf212, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf237, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf238, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf262, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf263, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf288, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf289, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf313, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf314, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf339, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf340, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf364, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf365, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf390, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf391, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf415, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf416, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf441, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf442, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf466, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf467, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf492, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf493, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf517, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf518, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf543, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf544, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf568, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf569, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf594, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), reinterpret_tensor(buf595, (1, 16, 128, 128), (262144, 128, 2048, 1), 0), primals_3, primals_10, primals_16, primals_23, primals_29, primals_36, primals_42, primals_49, primals_55, primals_62, primals_68, primals_75, primals_81, primals_88, primals_94, primals_101, primals_107, primals_114, primals_120, primals_127, primals_133, primals_140, primals_146, primals_153, primals_159, primals_166, primals_172, primals_179, primals_185, primals_192, primals_198, primals_205, primals_211, primals_218, primals_224, primals_231, primals_237, primals_244, primals_250, primals_257, primals_263, primals_270, primals_276, primals_283, primals_289, primals_296, primals_302, primals_309, primals_315, primals_343, primals_342, buf0, buf4, buf5, reinterpret_tensor(primals_318, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf15, buf20, buf21, buf22, buf23, buf24, buf30, buf31, reinterpret_tensor(primals_319, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf41, buf46, buf47, buf48, buf49, buf50, buf55, buf56, reinterpret_tensor(primals_320, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf66, buf71, buf72, buf73, buf74, buf75, buf81, buf82, reinterpret_tensor(primals_321, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf92, buf97, buf98, buf99, buf100, buf101, buf106, buf107, reinterpret_tensor(primals_322, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf117, buf122, buf123, buf124, buf125, buf126, buf132, buf133, reinterpret_tensor(primals_323, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf143, buf148, buf149, buf150, buf151, buf152, buf157, buf158, reinterpret_tensor(primals_324, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf168, buf173, buf174, buf175, buf176, buf177, buf183, buf184, reinterpret_tensor(primals_325, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf194, buf199, buf200, buf201, buf202, buf203, buf208, buf209, reinterpret_tensor(primals_326, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf219, buf224, buf225, buf226, buf227, buf228, buf234, buf235, reinterpret_tensor(primals_327, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf245, buf250, buf251, buf252, buf253, buf254, buf259, buf260, reinterpret_tensor(primals_328, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf270, buf275, buf276, buf277, buf278, buf279, buf285, buf286, reinterpret_tensor(primals_329, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf296, buf301, buf302, buf303, buf304, buf305, buf310, buf311, reinterpret_tensor(primals_330, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf321, buf326, buf327, buf328, buf329, buf330, buf336, buf337, reinterpret_tensor(primals_331, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf347, buf352, buf353, buf354, buf355, buf356, buf361, buf362, reinterpret_tensor(primals_332, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf372, buf377, buf378, buf379, buf380, buf381, buf387, buf388, reinterpret_tensor(primals_333, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf398, buf403, buf404, buf405, buf406, buf407, buf412, buf413, reinterpret_tensor(primals_334, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf423, buf428, buf429, buf430, buf431, buf432, buf438, buf439, reinterpret_tensor(primals_335, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf449, buf454, buf455, buf456, buf457, buf458, buf463, buf464, reinterpret_tensor(primals_336, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf474, buf479, buf480, buf481, buf482, buf483, buf489, buf490, reinterpret_tensor(primals_337, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf500, buf505, buf506, buf507, buf508, buf509, buf514, buf515, reinterpret_tensor(primals_338, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf525, buf530, buf531, buf532, buf533, buf534, buf540, buf541, reinterpret_tensor(primals_339, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf551, buf556, buf557, buf558, buf559, buf560, buf565, buf566, reinterpret_tensor(primals_340, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf576, buf581, buf582, buf583, buf584, buf585, buf591, buf592, reinterpret_tensor(primals_341, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf602, buf607, buf608, buf609, buf610, buf611, buf616, buf617, buf621, buf623, reinterpret_tensor(primals_317, (50257, 2048), (2048, 1), 0), buf625, reinterpret_tensor(primals_313, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_311, (8192, 2048), (2048, 1), 0), buf626, reinterpret_tensor(primals_307, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf600, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf595, (16, 128, 128), (128, 1, 2048), 0), buf627, reinterpret_tensor(buf593, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf594, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_306, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_305, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_304, (2048, 2048), (2048, 1), 0), buf628, reinterpret_tensor(primals_300, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_298, (8192, 2048), (2048, 1), 0), buf629, reinterpret_tensor(primals_294, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf574, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf569, (16, 128, 128), (128, 1, 2048), 0), buf630, reinterpret_tensor(buf567, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf568, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_293, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_292, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_291, (2048, 2048), (2048, 1), 0), buf631, reinterpret_tensor(primals_287, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_285, (8192, 2048), (2048, 1), 0), buf632, reinterpret_tensor(primals_281, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf549, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf544, (16, 128, 128), (128, 1, 2048), 0), buf633, reinterpret_tensor(buf542, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf543, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_280, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_279, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_278, (2048, 2048), (2048, 1), 0), buf634, reinterpret_tensor(primals_274, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_272, (8192, 2048), (2048, 1), 0), buf635, reinterpret_tensor(primals_268, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf523, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf518, (16, 128, 128), (128, 1, 2048), 0), buf636, reinterpret_tensor(buf516, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf517, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_267, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_266, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_265, (2048, 2048), (2048, 1), 0), buf637, reinterpret_tensor(primals_261, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_259, (8192, 2048), (2048, 1), 0), buf638, reinterpret_tensor(primals_255, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf498, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf493, (16, 128, 128), (128, 1, 2048), 0), buf639, reinterpret_tensor(buf491, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf492, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_254, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_253, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_252, (2048, 2048), (2048, 1), 0), buf640, reinterpret_tensor(primals_248, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_246, (8192, 2048), (2048, 1), 0), buf641, reinterpret_tensor(primals_242, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf472, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf467, (16, 128, 128), (128, 1, 2048), 0), buf642, reinterpret_tensor(buf465, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf466, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_241, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_240, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_239, (2048, 2048), (2048, 1), 0), buf643, reinterpret_tensor(primals_235, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_233, (8192, 2048), (2048, 1), 0), buf644, reinterpret_tensor(primals_229, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf447, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf442, (16, 128, 128), (128, 1, 2048), 0), buf645, reinterpret_tensor(buf440, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf441, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_228, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_227, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_226, (2048, 2048), (2048, 1), 0), buf646, reinterpret_tensor(primals_222, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_220, (8192, 2048), (2048, 1), 0), buf647, reinterpret_tensor(primals_216, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf421, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf416, (16, 128, 128), (128, 1, 2048), 0), buf648, reinterpret_tensor(buf414, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf415, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_215, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_214, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_213, (2048, 2048), (2048, 1), 0), buf649, reinterpret_tensor(primals_209, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_207, (8192, 2048), (2048, 1), 0), buf650, reinterpret_tensor(primals_203, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf396, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf391, (16, 128, 128), (128, 1, 2048), 0), buf651, reinterpret_tensor(buf389, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf390, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_202, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_201, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_200, (2048, 2048), (2048, 1), 0), buf652, reinterpret_tensor(primals_196, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_194, (8192, 2048), (2048, 1), 0), buf653, reinterpret_tensor(primals_190, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf370, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf365, (16, 128, 128), (128, 1, 2048), 0), buf654, reinterpret_tensor(buf363, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf364, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_189, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_188, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_187, (2048, 2048), (2048, 1), 0), buf655, reinterpret_tensor(primals_183, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_181, (8192, 2048), (2048, 1), 0), buf656, reinterpret_tensor(primals_177, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf345, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf340, (16, 128, 128), (128, 1, 2048), 0), buf657, reinterpret_tensor(buf338, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf339, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_176, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_175, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_174, (2048, 2048), (2048, 1), 0), buf658, reinterpret_tensor(primals_170, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_168, (8192, 2048), (2048, 1), 0), buf659, reinterpret_tensor(primals_164, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf319, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf314, (16, 128, 128), (128, 1, 2048), 0), buf660, reinterpret_tensor(buf312, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf313, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_163, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_162, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_161, (2048, 2048), (2048, 1), 0), buf661, reinterpret_tensor(primals_157, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_155, (8192, 2048), (2048, 1), 0), buf662, reinterpret_tensor(primals_151, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf294, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf289, (16, 128, 128), (128, 1, 2048), 0), buf663, reinterpret_tensor(buf287, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf288, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_150, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_149, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_148, (2048, 2048), (2048, 1), 0), buf664, reinterpret_tensor(primals_144, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_142, (8192, 2048), (2048, 1), 0), buf665, reinterpret_tensor(primals_138, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf268, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf263, (16, 128, 128), (128, 1, 2048), 0), buf666, reinterpret_tensor(buf261, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf262, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_137, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_136, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_135, (2048, 2048), (2048, 1), 0), buf667, reinterpret_tensor(primals_131, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_129, (8192, 2048), (2048, 1), 0), buf668, reinterpret_tensor(primals_125, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf243, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf238, (16, 128, 128), (128, 1, 2048), 0), buf669, reinterpret_tensor(buf236, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf237, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_124, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_123, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_122, (2048, 2048), (2048, 1), 0), buf670, reinterpret_tensor(primals_118, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_116, (8192, 2048), (2048, 1), 0), buf671, reinterpret_tensor(primals_112, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf217, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf212, (16, 128, 128), (128, 1, 2048), 0), buf672, reinterpret_tensor(buf210, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf211, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_111, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_110, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_109, (2048, 2048), (2048, 1), 0), buf673, reinterpret_tensor(primals_105, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_103, (8192, 2048), (2048, 1), 0), buf674, reinterpret_tensor(primals_99, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf192, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf187, (16, 128, 128), (128, 1, 2048), 0), buf675, reinterpret_tensor(buf185, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf186, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_98, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_97, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_96, (2048, 2048), (2048, 1), 0), buf676, reinterpret_tensor(primals_92, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_90, (8192, 2048), (2048, 1), 0), buf677, reinterpret_tensor(primals_86, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf166, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf161, (16, 128, 128), (128, 1, 2048), 0), buf678, reinterpret_tensor(buf159, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf160, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_85, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_84, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_83, (2048, 2048), (2048, 1), 0), buf679, reinterpret_tensor(primals_79, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_77, (8192, 2048), (2048, 1), 0), buf680, reinterpret_tensor(primals_73, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf141, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf136, (16, 128, 128), (128, 1, 2048), 0), buf681, reinterpret_tensor(buf134, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf135, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_72, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_71, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_70, (2048, 2048), (2048, 1), 0), buf682, reinterpret_tensor(primals_66, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_64, (8192, 2048), (2048, 1), 0), buf683, reinterpret_tensor(primals_60, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf115, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf110, (16, 128, 128), (128, 1, 2048), 0), buf684, reinterpret_tensor(buf108, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf109, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_59, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_58, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_57, (2048, 2048), (2048, 1), 0), buf685, reinterpret_tensor(primals_53, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_51, (8192, 2048), (2048, 1), 0), buf686, reinterpret_tensor(primals_47, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf90, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf85, (16, 128, 128), (128, 1, 2048), 0), buf687, reinterpret_tensor(buf83, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf84, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_46, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_45, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_44, (2048, 2048), (2048, 1), 0), buf688, reinterpret_tensor(primals_40, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_38, (8192, 2048), (2048, 1), 0), buf689, reinterpret_tensor(primals_34, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf64, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf59, (16, 128, 128), (128, 1, 2048), 0), buf690, reinterpret_tensor(buf57, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf58, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_33, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_32, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_31, (2048, 2048), (2048, 1), 0), buf691, reinterpret_tensor(primals_27, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_25, (8192, 2048), (2048, 1), 0), buf692, reinterpret_tensor(primals_21, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf39, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf34, (16, 128, 128), (128, 1, 2048), 0), buf693, reinterpret_tensor(buf32, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf33, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_20, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_19, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_18, (2048, 2048), (2048, 1), 0), buf694, reinterpret_tensor(primals_14, (2048, 8192), (8192, 1), 0), reinterpret_tensor(primals_12, (8192, 2048), (2048, 1), 0), buf695, reinterpret_tensor(primals_8, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf13, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf8, (16, 128, 128), (128, 1, 2048), 0), buf696, reinterpret_tensor(buf6, (16, 128, 128), (128, 1, 2048), 0), reinterpret_tensor(buf7, (16, 128, 128), (128, 2048, 1), 0), reinterpret_tensor(primals_7, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_6, (2048, 2048), (2048, 1), 0), reinterpret_tensor(primals_5, (2048, 2048), (2048, 1), 0), buf697, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50257, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((2048, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((8192, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((8192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((2048, 8192), (8192, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((50257, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_319 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_320 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_321 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_322 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_323 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_324 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_325 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_326 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_327 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_328 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_329 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_330 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_331 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_332 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_333 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_334 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_335 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_336 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_337 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_338 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_339 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_340 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_341 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_342 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_343 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPTNeoForCausalLM', benchmark_compiled_module)
