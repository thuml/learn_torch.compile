
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


cpp_fused_embedding_native_layer_norm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 50400);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 50400L), "index out of bounds: 0 <= tmp3 < 50400L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*tmp3)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 50400);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 50400L), "index out of bounds: 0 <= tmp3 < 50400L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*tmp3)));
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_3 = async_compile.cpp('''
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


cpp_fused_clone_4 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const long* in_ptr2,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = in_ptr2[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = decltype(tmp3)(tmp3 + 50400);
                        auto tmp5 = tmp3 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp3;
                        TORCH_CHECK((0 <= tmp6) & (tmp6 < 50400L), "index out of bounds: 0 <= tmp6 < 50400L")
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*tmp6)));
                        auto tmp8 = tmp2 + tmp7;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = in_ptr2[static_cast<long>(x0)];
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = decltype(tmp3)(tmp3 + 50400);
                    auto tmp5 = tmp3 < 0;
                    auto tmp6 = tmp5 ? tmp4 : tmp3;
                    TORCH_CHECK((0 <= tmp6) & (tmp6 < 50400L), "index out of bounds: 0 <= tmp6 < 50400L")
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (4096L*tmp6)));
                    auto tmp8 = tmp2 + tmp7;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 - tmp10;
                    auto tmp13 = static_cast<float>(4096.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = static_cast<float>(1e-05);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = 1 / std::sqrt(tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 * tmp18;
                    auto tmp21 = tmp19 * tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_9 = async_compile.cpp('''
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


cpp_fused_clone_10 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_embedding_native_layer_norm_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp6 = in_ptr3[static_cast<long>(x0)];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = decltype(tmp6)(tmp6 + 50400);
                        auto tmp8 = tmp6 < 0;
                        auto tmp9 = tmp8 ? tmp7 : tmp6;
                        TORCH_CHECK((0 <= tmp9) & (tmp9 < 50400L), "index out of bounds: 0 <= tmp9 < 50400L")
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (4096L*tmp9)));
                        auto tmp11 = tmp5 + tmp10;
                        auto tmp12 = tmp2 + tmp11;
                        tmp12.store(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp12);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_15 = async_compile.cpp('''
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


cpp_fused_clone_16 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_21 = async_compile.cpp('''
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


cpp_fused_clone_22 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_27 = async_compile.cpp('''
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


cpp_fused_clone_28 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_33 = async_compile.cpp('''
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


cpp_fused_clone_34 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_39 = async_compile.cpp('''
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


cpp_fused_clone_40 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_45 = async_compile.cpp('''
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


cpp_fused_clone_46 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_51 = async_compile.cpp('''
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


cpp_fused_clone_52 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_57 = async_compile.cpp('''
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


cpp_fused_clone_58 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_63 = async_compile.cpp('''
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


cpp_fused_clone_64 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_69 = async_compile.cpp('''
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


cpp_fused_clone_70 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_75 = async_compile.cpp('''
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


cpp_fused_clone_76 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_81 = async_compile.cpp('''
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


cpp_fused_clone_82 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_87 = async_compile.cpp('''
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


cpp_fused_clone_88 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_93 = async_compile.cpp('''
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


cpp_fused_clone_94 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_99 = async_compile.cpp('''
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


cpp_fused_clone_100 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_105 = async_compile.cpp('''
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


cpp_fused_clone_106 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_111 = async_compile.cpp('''
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


cpp_fused_clone_112 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_117 = async_compile.cpp('''
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


cpp_fused_clone_118 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_121 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_123 = async_compile.cpp('''
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


cpp_fused_clone_124 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_127 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_129 = async_compile.cpp('''
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


cpp_fused_clone_130 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_133 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_135 = async_compile.cpp('''
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


cpp_fused_clone_136 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_139 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_141 = async_compile.cpp('''
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


cpp_fused_clone_142 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_145 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_146 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_147 = async_compile.cpp('''
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


cpp_fused_clone_148 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_149 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_150 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_151 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_152 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_153 = async_compile.cpp('''
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


cpp_fused_clone_154 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_155 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_156 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_157 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_158 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_159 = async_compile.cpp('''
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


cpp_fused_clone_160 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_161 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_162 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(4096.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_163 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(64);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp10 = static_cast<long>(0);
                            auto tmp11 = tmp9 >= tmp10;
                            auto tmp12 = static_cast<long>(1);
                            auto tmp13 = tmp9 < tmp12;
                            auto tmp14 = [&]
                            {
                                auto tmp15 = in_ptr0[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp16 = decltype(tmp15)(-tmp15);
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp13 ? tmp14() : static_cast<decltype(tmp14())>(0.0);
                            auto tmp18 = tmp9 >= tmp12;
                            auto tmp19 = static_cast<long>(2);
                            auto tmp20 = tmp9 < tmp19;
                            auto tmp21 = [&]
                            {
                                auto tmp22 = in_ptr0[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp22;
                            }
                            ;
                            auto tmp23 = tmp18 ? tmp21() : static_cast<decltype(tmp21())>(0.0);
                            auto tmp24 = tmp13 ? tmp17 : tmp23;
                            auto tmp25 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 + tmp26);
                            return tmp27;
                        }
                        ;
                        auto tmp28 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp29 = tmp0 >= tmp3;
                        auto tmp30 = static_cast<long>(256);
                        auto tmp31 = tmp0 < tmp30;
                        auto tmp32 = [&]
                        {
                            auto tmp33 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp33;
                        }
                        ;
                        auto tmp34 = tmp29 ? tmp32() : static_cast<decltype(tmp32())>(0.0);
                        auto tmp35 = tmp4 ? tmp28 : tmp34;
                        auto tmp36 = [&]
                        {
                            auto tmp37 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            auto tmp38 = in_ptr1[static_cast<long>(32L + (64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = c10::convert<long>(static_cast<long>(x2) % static_cast<long>(2L));
                            auto tmp41 = static_cast<long>(0);
                            auto tmp42 = tmp40 >= tmp41;
                            auto tmp43 = static_cast<long>(1);
                            auto tmp44 = tmp40 < tmp43;
                            auto tmp45 = [&]
                            {
                                auto tmp46 = in_ptr2[static_cast<long>(1L + (2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                auto tmp47 = decltype(tmp46)(-tmp46);
                                return tmp47;
                            }
                            ;
                            auto tmp48 = tmp44 ? tmp45() : static_cast<decltype(tmp45())>(0.0);
                            auto tmp49 = tmp40 >= tmp43;
                            auto tmp50 = static_cast<long>(2);
                            auto tmp51 = tmp40 < tmp50;
                            auto tmp52 = [&]
                            {
                                auto tmp53 = in_ptr2[static_cast<long>((2L*(static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L))) + (256L*x1) + (4096L*x0))];
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp49 ? tmp52() : static_cast<decltype(tmp52())>(0.0);
                            auto tmp55 = tmp44 ? tmp48 : tmp54;
                            auto tmp56 = in_ptr1[static_cast<long>((64L*x0) + (static_cast<long>(c10::div_floor_integer(x2, 2L)) % static_cast<long>(32L)))];
                            auto tmp57 = decltype(tmp55)(tmp55 * tmp56);
                            auto tmp58 = decltype(tmp39)(tmp39 + tmp57);
                            return tmp58;
                        }
                        ;
                        auto tmp59 = tmp4 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                        auto tmp60 = [&]
                        {
                            auto tmp61 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (4096L*x0))];
                            return tmp61;
                        }
                        ;
                        auto tmp62 = tmp29 ? tmp60() : static_cast<decltype(tmp60())>(0.0);
                        auto tmp63 = tmp4 ? tmp59 : tmp62;
                        out_ptr0[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp35;
                        out_ptr1[static_cast<long>(x2 + (256L*x1) + (4096L*x0))] = tmp63;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_lift_fresh_where_164 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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
                            auto tmp5 = in_ptr2[static_cast<long>(0L)];
                            auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp7);
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
                        auto tmp5 = in_ptr2[static_cast<long>(0L)];
                        auto tmp8 = out_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp2 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = decltype(tmp1)::blendv(tmp3, tmp1, tmp0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 - tmp9;
                        auto tmp11 = tmp10.exp();
                        tmp11.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (16384L*x0)));
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


cpp_fused__softmax_165 = async_compile.cpp('''
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


cpp_fused_clone_166 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0) + (32768L*x1)));
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_167 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_168 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(524288L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp5 = tmp3 + tmp4;
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp2 + tmp7;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(4096.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_169 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const long* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr1 = in_out_ptr0;
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr1[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = out_ptr1[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr2[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
            float tmp_acc0 = -std::numeric_limits<float>::infinity();
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>(1L + (2L*x0) + (2L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                tmp0.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
            }
            tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
            out_ptr4[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        {
            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
            float tmp_acc0 = 0;
            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = out_ptr4[static_cast<long>(0L)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 - tmp2;
                auto tmp4 = tmp3.exp();
                tmp_acc0_vec = tmp_acc0_vec + tmp4;
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr5[static_cast<long>(0L)] = static_cast<float>(tmp_acc0);
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp11 = out_ptr1[static_cast<long>(0L)];
        auto tmp13 = out_ptr2[static_cast<long>(0L)];
        auto tmp22 = in_ptr2[static_cast<long>(0L)];
        auto tmp31 = out_ptr4[static_cast<long>(0L)];
        auto tmp33 = out_ptr5[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(0);
        auto tmp2 = max_propagate_nan(tmp0, tmp1);
        auto tmp3 = static_cast<long>(128);
        auto tmp4 = min_propagate_nan(tmp2, tmp3);
        auto tmp5 = tmp4 != tmp3;
        auto tmp6 = tmp5 ? tmp4 : tmp1;
        auto tmp7 = decltype(tmp6)(tmp6 + 128);
        auto tmp8 = tmp6 < 0;
        auto tmp9 = tmp8 ? tmp7 : tmp6;
        TORCH_CHECK((0 <= tmp9) & (tmp9 < 128L), "index out of bounds: 0 <= tmp9 < 128L")
        auto tmp10 = out_ptr0[static_cast<long>(tmp9)];
        auto tmp12 = decltype(tmp10)(tmp10 - tmp11);
        auto tmp14 = std::log(tmp13);
        auto tmp15 = decltype(tmp12)(tmp12 - tmp14);
        auto tmp16 = decltype(tmp15)(-tmp15);
        auto tmp17 = static_cast<float>(0.0);
        auto tmp18 = tmp5 ? tmp16 : tmp17;
        auto tmp19 = c10::convert<long>(tmp5);
        auto tmp20 = c10::convert<float>(tmp19);
        auto tmp21 = tmp18 / tmp20;
        auto tmp23 = max_propagate_nan(tmp22, tmp1);
        auto tmp24 = min_propagate_nan(tmp23, tmp3);
        auto tmp25 = tmp24 != tmp3;
        auto tmp26 = tmp25 ? tmp24 : tmp1;
        auto tmp27 = decltype(tmp26)(tmp26 + 128);
        auto tmp28 = tmp26 < 0;
        auto tmp29 = tmp28 ? tmp27 : tmp26;
        TORCH_CHECK((0 <= tmp29) & (tmp29 < 128L), "index out of bounds: 0 <= tmp29 < 128L")
        auto tmp30 = out_ptr3[static_cast<long>(tmp29)];
        auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
        auto tmp34 = std::log(tmp33);
        auto tmp35 = decltype(tmp32)(tmp32 - tmp34);
        auto tmp36 = decltype(tmp35)(-tmp35);
        auto tmp37 = tmp25 ? tmp36 : tmp17;
        auto tmp38 = c10::convert<long>(tmp25);
        auto tmp39 = c10::convert<float>(tmp38);
        auto tmp40 = tmp37 / tmp39;
        auto tmp41 = decltype(tmp21)(tmp21 + tmp40);
        auto tmp42 = static_cast<float>(2.0);
        auto tmp43 = tmp41 / tmp42;
        in_out_ptr0[static_cast<long>(0L)] = tmp43;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1 = args
    args.clear()
    assert_size_stride(arg0_1, (50400, 4096), (4096, 1))
    assert_size_stride(arg1_1, (4096, ), (1, ))
    assert_size_stride(arg2_1, (4096, ), (1, ))
    assert_size_stride(arg3_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg4_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg5_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg6_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg7_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg8_1, (16384, ), (1, ))
    assert_size_stride(arg9_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg10_1, (4096, ), (1, ))
    assert_size_stride(arg11_1, (4096, ), (1, ))
    assert_size_stride(arg12_1, (4096, ), (1, ))
    assert_size_stride(arg13_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg14_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg15_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg16_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg17_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg18_1, (16384, ), (1, ))
    assert_size_stride(arg19_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg20_1, (4096, ), (1, ))
    assert_size_stride(arg21_1, (4096, ), (1, ))
    assert_size_stride(arg22_1, (4096, ), (1, ))
    assert_size_stride(arg23_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg24_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg25_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg26_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg27_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg28_1, (16384, ), (1, ))
    assert_size_stride(arg29_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg30_1, (4096, ), (1, ))
    assert_size_stride(arg31_1, (4096, ), (1, ))
    assert_size_stride(arg32_1, (4096, ), (1, ))
    assert_size_stride(arg33_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg34_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg35_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg36_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg37_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg38_1, (16384, ), (1, ))
    assert_size_stride(arg39_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg40_1, (4096, ), (1, ))
    assert_size_stride(arg41_1, (4096, ), (1, ))
    assert_size_stride(arg42_1, (4096, ), (1, ))
    assert_size_stride(arg43_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg44_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg45_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg46_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg47_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg48_1, (16384, ), (1, ))
    assert_size_stride(arg49_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg50_1, (4096, ), (1, ))
    assert_size_stride(arg51_1, (4096, ), (1, ))
    assert_size_stride(arg52_1, (4096, ), (1, ))
    assert_size_stride(arg53_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg54_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg55_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg56_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg57_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg58_1, (16384, ), (1, ))
    assert_size_stride(arg59_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg60_1, (4096, ), (1, ))
    assert_size_stride(arg61_1, (4096, ), (1, ))
    assert_size_stride(arg62_1, (4096, ), (1, ))
    assert_size_stride(arg63_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg64_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg65_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg66_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg67_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg68_1, (16384, ), (1, ))
    assert_size_stride(arg69_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg70_1, (4096, ), (1, ))
    assert_size_stride(arg71_1, (4096, ), (1, ))
    assert_size_stride(arg72_1, (4096, ), (1, ))
    assert_size_stride(arg73_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg74_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg75_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg76_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg77_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg78_1, (16384, ), (1, ))
    assert_size_stride(arg79_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg80_1, (4096, ), (1, ))
    assert_size_stride(arg81_1, (4096, ), (1, ))
    assert_size_stride(arg82_1, (4096, ), (1, ))
    assert_size_stride(arg83_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg84_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg85_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg86_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg87_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg88_1, (16384, ), (1, ))
    assert_size_stride(arg89_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg90_1, (4096, ), (1, ))
    assert_size_stride(arg91_1, (4096, ), (1, ))
    assert_size_stride(arg92_1, (4096, ), (1, ))
    assert_size_stride(arg93_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg94_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg95_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg96_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg97_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg98_1, (16384, ), (1, ))
    assert_size_stride(arg99_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg100_1, (4096, ), (1, ))
    assert_size_stride(arg101_1, (4096, ), (1, ))
    assert_size_stride(arg102_1, (4096, ), (1, ))
    assert_size_stride(arg103_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg104_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg105_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg106_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg107_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg108_1, (16384, ), (1, ))
    assert_size_stride(arg109_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg110_1, (4096, ), (1, ))
    assert_size_stride(arg111_1, (4096, ), (1, ))
    assert_size_stride(arg112_1, (4096, ), (1, ))
    assert_size_stride(arg113_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg114_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg115_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg116_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg117_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg118_1, (16384, ), (1, ))
    assert_size_stride(arg119_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg120_1, (4096, ), (1, ))
    assert_size_stride(arg121_1, (4096, ), (1, ))
    assert_size_stride(arg122_1, (4096, ), (1, ))
    assert_size_stride(arg123_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg124_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg125_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg126_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg127_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg128_1, (16384, ), (1, ))
    assert_size_stride(arg129_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg130_1, (4096, ), (1, ))
    assert_size_stride(arg131_1, (4096, ), (1, ))
    assert_size_stride(arg132_1, (4096, ), (1, ))
    assert_size_stride(arg133_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg134_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg135_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg136_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg137_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg138_1, (16384, ), (1, ))
    assert_size_stride(arg139_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg140_1, (4096, ), (1, ))
    assert_size_stride(arg141_1, (4096, ), (1, ))
    assert_size_stride(arg142_1, (4096, ), (1, ))
    assert_size_stride(arg143_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg144_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg145_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg146_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg147_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg148_1, (16384, ), (1, ))
    assert_size_stride(arg149_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg150_1, (4096, ), (1, ))
    assert_size_stride(arg151_1, (4096, ), (1, ))
    assert_size_stride(arg152_1, (4096, ), (1, ))
    assert_size_stride(arg153_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg154_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg155_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg156_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg157_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg158_1, (16384, ), (1, ))
    assert_size_stride(arg159_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg160_1, (4096, ), (1, ))
    assert_size_stride(arg161_1, (4096, ), (1, ))
    assert_size_stride(arg162_1, (4096, ), (1, ))
    assert_size_stride(arg163_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg164_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg165_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg166_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg167_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg168_1, (16384, ), (1, ))
    assert_size_stride(arg169_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg170_1, (4096, ), (1, ))
    assert_size_stride(arg171_1, (4096, ), (1, ))
    assert_size_stride(arg172_1, (4096, ), (1, ))
    assert_size_stride(arg173_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg174_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg175_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg176_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg177_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg178_1, (16384, ), (1, ))
    assert_size_stride(arg179_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg180_1, (4096, ), (1, ))
    assert_size_stride(arg181_1, (4096, ), (1, ))
    assert_size_stride(arg182_1, (4096, ), (1, ))
    assert_size_stride(arg183_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg184_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg185_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg186_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg187_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg188_1, (16384, ), (1, ))
    assert_size_stride(arg189_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg190_1, (4096, ), (1, ))
    assert_size_stride(arg191_1, (4096, ), (1, ))
    assert_size_stride(arg192_1, (4096, ), (1, ))
    assert_size_stride(arg193_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg194_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg195_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg196_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg197_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg198_1, (16384, ), (1, ))
    assert_size_stride(arg199_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg200_1, (4096, ), (1, ))
    assert_size_stride(arg201_1, (4096, ), (1, ))
    assert_size_stride(arg202_1, (4096, ), (1, ))
    assert_size_stride(arg203_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg204_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg205_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg206_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg207_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg208_1, (16384, ), (1, ))
    assert_size_stride(arg209_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg210_1, (4096, ), (1, ))
    assert_size_stride(arg211_1, (4096, ), (1, ))
    assert_size_stride(arg212_1, (4096, ), (1, ))
    assert_size_stride(arg213_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg214_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg215_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg216_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg217_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg218_1, (16384, ), (1, ))
    assert_size_stride(arg219_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg220_1, (4096, ), (1, ))
    assert_size_stride(arg221_1, (4096, ), (1, ))
    assert_size_stride(arg222_1, (4096, ), (1, ))
    assert_size_stride(arg223_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg224_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg225_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg226_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg227_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg228_1, (16384, ), (1, ))
    assert_size_stride(arg229_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg230_1, (4096, ), (1, ))
    assert_size_stride(arg231_1, (4096, ), (1, ))
    assert_size_stride(arg232_1, (4096, ), (1, ))
    assert_size_stride(arg233_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg234_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg235_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg236_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg237_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg238_1, (16384, ), (1, ))
    assert_size_stride(arg239_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg240_1, (4096, ), (1, ))
    assert_size_stride(arg241_1, (4096, ), (1, ))
    assert_size_stride(arg242_1, (4096, ), (1, ))
    assert_size_stride(arg243_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg244_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg245_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg246_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg247_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg248_1, (16384, ), (1, ))
    assert_size_stride(arg249_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg250_1, (4096, ), (1, ))
    assert_size_stride(arg251_1, (4096, ), (1, ))
    assert_size_stride(arg252_1, (4096, ), (1, ))
    assert_size_stride(arg253_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg254_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg255_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg256_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg257_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg258_1, (16384, ), (1, ))
    assert_size_stride(arg259_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg260_1, (4096, ), (1, ))
    assert_size_stride(arg261_1, (4096, ), (1, ))
    assert_size_stride(arg262_1, (4096, ), (1, ))
    assert_size_stride(arg263_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg264_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg265_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg266_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg267_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg268_1, (16384, ), (1, ))
    assert_size_stride(arg269_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg270_1, (4096, ), (1, ))
    assert_size_stride(arg271_1, (4096, ), (1, ))
    assert_size_stride(arg272_1, (4096, ), (1, ))
    assert_size_stride(arg273_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg274_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg275_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg276_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg277_1, (16384, 4096), (4096, 1))
    assert_size_stride(arg278_1, (16384, ), (1, ))
    assert_size_stride(arg279_1, (4096, 16384), (16384, 1))
    assert_size_stride(arg280_1, (4096, ), (1, ))
    assert_size_stride(arg281_1, (4096, ), (1, ))
    assert_size_stride(arg282_1, (4096, ), (1, ))
    assert_size_stride(arg283_1, (2, 4096), (4096, 1))
    assert_size_stride(arg284_1, (2, ), (1, ))
    assert_size_stride(arg285_1, (2048, 64), (64, 1))
    assert_size_stride(arg286_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg287_1, (), ())
    assert_size_stride(arg288_1, (2048, 64), (64, 1))
    assert_size_stride(arg289_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg290_1, (), ())
    assert_size_stride(arg291_1, (2048, 64), (64, 1))
    assert_size_stride(arg292_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg293_1, (), ())
    assert_size_stride(arg294_1, (2048, 64), (64, 1))
    assert_size_stride(arg295_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg296_1, (), ())
    assert_size_stride(arg297_1, (2048, 64), (64, 1))
    assert_size_stride(arg298_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg299_1, (), ())
    assert_size_stride(arg300_1, (2048, 64), (64, 1))
    assert_size_stride(arg301_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg302_1, (), ())
    assert_size_stride(arg303_1, (2048, 64), (64, 1))
    assert_size_stride(arg304_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg305_1, (), ())
    assert_size_stride(arg306_1, (2048, 64), (64, 1))
    assert_size_stride(arg307_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg308_1, (), ())
    assert_size_stride(arg309_1, (2048, 64), (64, 1))
    assert_size_stride(arg310_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg311_1, (), ())
    assert_size_stride(arg312_1, (2048, 64), (64, 1))
    assert_size_stride(arg313_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg314_1, (), ())
    assert_size_stride(arg315_1, (2048, 64), (64, 1))
    assert_size_stride(arg316_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg317_1, (), ())
    assert_size_stride(arg318_1, (2048, 64), (64, 1))
    assert_size_stride(arg319_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg320_1, (), ())
    assert_size_stride(arg321_1, (2048, 64), (64, 1))
    assert_size_stride(arg322_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg323_1, (), ())
    assert_size_stride(arg324_1, (2048, 64), (64, 1))
    assert_size_stride(arg325_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg326_1, (), ())
    assert_size_stride(arg327_1, (2048, 64), (64, 1))
    assert_size_stride(arg328_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg329_1, (), ())
    assert_size_stride(arg330_1, (2048, 64), (64, 1))
    assert_size_stride(arg331_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg332_1, (), ())
    assert_size_stride(arg333_1, (2048, 64), (64, 1))
    assert_size_stride(arg334_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg335_1, (), ())
    assert_size_stride(arg336_1, (2048, 64), (64, 1))
    assert_size_stride(arg337_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg338_1, (), ())
    assert_size_stride(arg339_1, (2048, 64), (64, 1))
    assert_size_stride(arg340_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg341_1, (), ())
    assert_size_stride(arg342_1, (2048, 64), (64, 1))
    assert_size_stride(arg343_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg344_1, (), ())
    assert_size_stride(arg345_1, (2048, 64), (64, 1))
    assert_size_stride(arg346_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg347_1, (), ())
    assert_size_stride(arg348_1, (2048, 64), (64, 1))
    assert_size_stride(arg349_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg350_1, (), ())
    assert_size_stride(arg351_1, (2048, 64), (64, 1))
    assert_size_stride(arg352_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg353_1, (), ())
    assert_size_stride(arg354_1, (2048, 64), (64, 1))
    assert_size_stride(arg355_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg356_1, (), ())
    assert_size_stride(arg357_1, (2048, 64), (64, 1))
    assert_size_stride(arg358_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg359_1, (), ())
    assert_size_stride(arg360_1, (2048, 64), (64, 1))
    assert_size_stride(arg361_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg362_1, (), ())
    assert_size_stride(arg363_1, (2048, 64), (64, 1))
    assert_size_stride(arg364_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg365_1, (), ())
    assert_size_stride(arg366_1, (2048, 64), (64, 1))
    assert_size_stride(arg367_1, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(arg368_1, (), ())
    assert_size_stride(arg369_1, (1, 128), (128, 1))
    assert_size_stride(arg370_1, (1, ), (1, ))
    assert_size_stride(arg371_1, (1, ), (1, ))
    buf0 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf3 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_native_layer_norm_0(c_void_p(arg369_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg1_1
    del arg2_1
    buf4 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg3_1, (4096, 4096), (1, 4096), 0), out=buf4)
    del arg3_1
    buf5 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg4_1, (4096, 4096), (1, 4096), 0), out=buf5)
    del arg4_1
    buf6 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_1(c_void_p(buf4.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg285_1
    buf8 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf6, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf7, (16, 256, 128), (256, 1, 4096), 0), out=buf8)
    buf9 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf10 = reinterpret_tensor(buf8, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf8  # reuse
    buf11 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_div_lift_fresh_where_2(c_void_p(buf10.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()))
    del arg286_1
    del arg287_1
    buf12 = reinterpret_tensor(buf7, (128, 4096), (4096, 1), 0); del buf7  # reuse
    # Source Nodes: [value], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg5_1, (4096, 4096), (1, 4096), 0), out=buf12)
    del arg5_1
    buf13 = buf10; del buf10  # reuse
    cpp_fused__softmax_3(c_void_p(buf13.data_ptr()), c_void_p(buf11.data_ptr()))
    buf14 = reinterpret_tensor(buf6, (16, 128, 256), (32768, 256, 1), 0); del buf6  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf13, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf12, (16, 128, 256), (256, 4096, 1), 0), out=buf14)
    buf15 = reinterpret_tensor(buf12, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf12  # reuse
    cpp_fused_clone_4(c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf14, (128, 4096), (4096, 1), 0); del buf14  # reuse
    # Source Nodes: [attn_output_2], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg6_1, (4096, 4096), (1, 4096), 0), out=buf16)
    del arg6_1
    buf17 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg8_1, reinterpret_tensor(buf3, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg7_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf17)
    del arg7_1
    del arg8_1
    buf18 = reinterpret_tensor(buf17, (1, 128, 16384), (2097152, 16384, 1), 0); del buf17  # reuse
    cpp_fused_add_mul_pow_tanh_5(c_void_p(buf18.data_ptr()))
    buf19 = reinterpret_tensor(buf3, (128, 4096), (4096, 1), 0); del buf3  # reuse
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg10_1, reinterpret_tensor(buf18, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg9_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf19)
    del arg10_1
    del arg9_1
    buf20 = buf1; del buf1  # reuse
    buf21 = buf0; del buf0  # reuse
    buf23 = reinterpret_tensor(buf15, (1, 128, 4096), (524288, 4096, 1), 0); del buf15  # reuse
    cpp_fused_add_embedding_native_layer_norm_6(c_void_p(buf16.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg11_1
    del arg12_1
    buf24 = buf5; del buf5  # reuse
    # Source Nodes: [query_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg13_1, (4096, 4096), (1, 4096), 0), out=buf24)
    del arg13_1
    buf25 = buf4; del buf4  # reuse
    # Source Nodes: [key_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg14_1, (4096, 4096), (1, 4096), 0), out=buf25)
    del arg14_1
    buf26 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    buf27 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_7(c_void_p(buf24.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del arg288_1
    buf28 = reinterpret_tensor(buf13, (16, 128, 128), (16384, 128, 1), 0); del buf13  # reuse
    # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf26, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf27, (16, 256, 128), (256, 1, 4096), 0), out=buf28)
    buf29 = buf11; del buf11  # reuse
    buf30 = reinterpret_tensor(buf28, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf28  # reuse
    buf31 = buf9; del buf9  # reuse
    cpp_fused__softmax_div_lift_fresh_where_8(c_void_p(buf30.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg289_1
    del arg290_1
    buf32 = reinterpret_tensor(buf27, (128, 4096), (4096, 1), 0); del buf27  # reuse
    # Source Nodes: [value_2], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg15_1, (4096, 4096), (1, 4096), 0), out=buf32)
    del arg15_1
    buf33 = buf30; del buf30  # reuse
    cpp_fused__softmax_9(c_void_p(buf33.data_ptr()), c_void_p(buf31.data_ptr()))
    buf34 = reinterpret_tensor(buf26, (16, 128, 256), (32768, 256, 1), 0); del buf26  # reuse
    # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf33, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf32, (16, 128, 256), (256, 4096, 1), 0), out=buf34)
    buf35 = reinterpret_tensor(buf32, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf32  # reuse
    cpp_fused_clone_10(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    buf36 = reinterpret_tensor(buf34, (128, 4096), (4096, 1), 0); del buf34  # reuse
    # Source Nodes: [attn_output_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf35, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg16_1, (4096, 4096), (1, 4096), 0), out=buf36)
    del arg16_1
    buf37 = reinterpret_tensor(buf18, (128, 16384), (16384, 1), 0); del buf18  # reuse
    # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg18_1, reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg17_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf37)
    del arg17_1
    del arg18_1
    buf38 = reinterpret_tensor(buf37, (1, 128, 16384), (2097152, 16384, 1), 0); del buf37  # reuse
    cpp_fused_add_mul_pow_tanh_11(c_void_p(buf38.data_ptr()))
    buf39 = reinterpret_tensor(buf23, (128, 4096), (4096, 1), 0); del buf23  # reuse
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg20_1, reinterpret_tensor(buf38, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg19_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf39)
    del arg19_1
    del arg20_1
    buf40 = reinterpret_tensor(buf36, (1, 128, 4096), (524288, 4096, 1), 0); del buf36  # reuse
    buf41 = buf21; del buf21  # reuse
    buf42 = buf20; del buf20  # reuse
    buf44 = reinterpret_tensor(buf35, (1, 128, 4096), (524288, 4096, 1), 0); del buf35  # reuse
    cpp_fused_add_embedding_native_layer_norm_12(c_void_p(buf40.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf44.data_ptr()))
    del arg0_1
    del arg21_1
    del arg22_1
    del arg369_1
    buf45 = buf39; del buf39  # reuse
    # Source Nodes: [query_10], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg23_1, (4096, 4096), (1, 4096), 0), out=buf45)
    del arg23_1
    buf46 = buf19; del buf19  # reuse
    # Source Nodes: [key_10], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg24_1, (4096, 4096), (1, 4096), 0), out=buf46)
    del arg24_1
    buf47 = reinterpret_tensor(buf16, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf16  # reuse
    buf48 = reinterpret_tensor(buf25, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf25  # reuse
    cpp_fused_cat_13(c_void_p(buf45.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg291_1
    buf49 = reinterpret_tensor(buf33, (16, 128, 128), (16384, 128, 1), 0); del buf33  # reuse
    # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf47, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf48, (16, 256, 128), (256, 1, 4096), 0), out=buf49)
    buf50 = buf31; del buf31  # reuse
    buf51 = reinterpret_tensor(buf49, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf49  # reuse
    buf52 = buf29; del buf29  # reuse
    cpp_fused__softmax_div_lift_fresh_where_14(c_void_p(buf51.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()))
    del arg292_1
    del arg293_1
    buf53 = reinterpret_tensor(buf48, (128, 4096), (4096, 1), 0); del buf48  # reuse
    # Source Nodes: [value_4], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf44, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg25_1, (4096, 4096), (1, 4096), 0), out=buf53)
    del arg25_1
    buf54 = buf51; del buf51  # reuse
    cpp_fused__softmax_15(c_void_p(buf54.data_ptr()), c_void_p(buf52.data_ptr()))
    buf55 = reinterpret_tensor(buf47, (16, 128, 256), (32768, 256, 1), 0); del buf47  # reuse
    # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf54, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf53, (16, 128, 256), (256, 4096, 1), 0), out=buf55)
    buf56 = reinterpret_tensor(buf53, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf53  # reuse
    cpp_fused_clone_16(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = reinterpret_tensor(buf55, (128, 4096), (4096, 1), 0); del buf55  # reuse
    # Source Nodes: [attn_output_14], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf56, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg26_1, (4096, 4096), (1, 4096), 0), out=buf57)
    del arg26_1
    buf58 = reinterpret_tensor(buf38, (128, 16384), (16384, 1), 0); del buf38  # reuse
    # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg28_1, reinterpret_tensor(buf44, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg27_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf58)
    del arg27_1
    del arg28_1
    buf59 = reinterpret_tensor(buf58, (1, 128, 16384), (2097152, 16384, 1), 0); del buf58  # reuse
    cpp_fused_add_mul_pow_tanh_17(c_void_p(buf59.data_ptr()))
    buf60 = reinterpret_tensor(buf44, (128, 4096), (4096, 1), 0); del buf44  # reuse
    # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg30_1, reinterpret_tensor(buf59, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg29_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf60)
    del arg29_1
    del arg30_1
    buf61 = buf42; del buf42  # reuse
    buf62 = buf41; del buf41  # reuse
    buf64 = reinterpret_tensor(buf56, (1, 128, 4096), (524288, 4096, 1), 0); del buf56  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf57.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg31_1
    del arg32_1
    buf65 = buf46; del buf46  # reuse
    # Source Nodes: [query_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg33_1, (4096, 4096), (1, 4096), 0), out=buf65)
    del arg33_1
    buf66 = buf45; del buf45  # reuse
    # Source Nodes: [key_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg34_1, (4096, 4096), (1, 4096), 0), out=buf66)
    del arg34_1
    buf67 = reinterpret_tensor(buf24, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf24  # reuse
    buf68 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_19(c_void_p(buf65.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del arg294_1
    buf69 = reinterpret_tensor(buf54, (16, 128, 128), (16384, 128, 1), 0); del buf54  # reuse
    # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf68, (16, 256, 128), (256, 1, 4096), 0), out=buf69)
    buf70 = buf52; del buf52  # reuse
    buf71 = reinterpret_tensor(buf69, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf69  # reuse
    buf72 = buf50; del buf50  # reuse
    cpp_fused__softmax_div_lift_fresh_where_20(c_void_p(buf71.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg295_1
    del arg296_1
    buf73 = reinterpret_tensor(buf68, (128, 4096), (4096, 1), 0); del buf68  # reuse
    # Source Nodes: [value_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf64, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg35_1, (4096, 4096), (1, 4096), 0), out=buf73)
    del arg35_1
    buf74 = buf71; del buf71  # reuse
    cpp_fused__softmax_21(c_void_p(buf74.data_ptr()), c_void_p(buf72.data_ptr()))
    buf75 = reinterpret_tensor(buf67, (16, 128, 256), (32768, 256, 1), 0); del buf67  # reuse
    # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf74, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf73, (16, 128, 256), (256, 4096, 1), 0), out=buf75)
    buf76 = reinterpret_tensor(buf73, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf73  # reuse
    cpp_fused_clone_22(c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = reinterpret_tensor(buf75, (128, 4096), (4096, 1), 0); del buf75  # reuse
    # Source Nodes: [attn_output_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg36_1, (4096, 4096), (1, 4096), 0), out=buf77)
    del arg36_1
    buf78 = reinterpret_tensor(buf59, (128, 16384), (16384, 1), 0); del buf59  # reuse
    # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf64, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg37_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf78)
    del arg37_1
    del arg38_1
    buf79 = reinterpret_tensor(buf78, (1, 128, 16384), (2097152, 16384, 1), 0); del buf78  # reuse
    cpp_fused_add_mul_pow_tanh_23(c_void_p(buf79.data_ptr()))
    buf80 = reinterpret_tensor(buf64, (128, 4096), (4096, 1), 0); del buf64  # reuse
    # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg40_1, reinterpret_tensor(buf79, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg39_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf80)
    del arg39_1
    del arg40_1
    buf81 = reinterpret_tensor(buf77, (1, 128, 4096), (524288, 4096, 1), 0); del buf77  # reuse
    buf82 = buf62; del buf62  # reuse
    buf83 = buf61; del buf61  # reuse
    buf85 = reinterpret_tensor(buf76, (1, 128, 4096), (524288, 4096, 1), 0); del buf76  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf81.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg41_1
    del arg42_1
    buf86 = buf80; del buf80  # reuse
    # Source Nodes: [query_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg43_1, (4096, 4096), (1, 4096), 0), out=buf86)
    del arg43_1
    buf87 = buf60; del buf60  # reuse
    # Source Nodes: [key_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg44_1, (4096, 4096), (1, 4096), 0), out=buf87)
    del arg44_1
    buf88 = reinterpret_tensor(buf57, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf57  # reuse
    buf89 = reinterpret_tensor(buf40, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf40  # reuse
    cpp_fused_cat_25(c_void_p(buf86.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg297_1
    buf90 = reinterpret_tensor(buf74, (16, 128, 128), (16384, 128, 1), 0); del buf74  # reuse
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf89, (16, 256, 128), (256, 1, 4096), 0), out=buf90)
    buf91 = buf72; del buf72  # reuse
    buf92 = reinterpret_tensor(buf90, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf90  # reuse
    buf93 = buf70; del buf70  # reuse
    cpp_fused__softmax_div_lift_fresh_where_26(c_void_p(buf92.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()))
    del arg298_1
    del arg299_1
    buf94 = reinterpret_tensor(buf89, (128, 4096), (4096, 1), 0); del buf89  # reuse
    # Source Nodes: [value_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf85, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg45_1, (4096, 4096), (1, 4096), 0), out=buf94)
    del arg45_1
    buf95 = buf92; del buf92  # reuse
    cpp_fused__softmax_27(c_void_p(buf95.data_ptr()), c_void_p(buf93.data_ptr()))
    buf96 = reinterpret_tensor(buf88, (16, 128, 256), (32768, 256, 1), 0); del buf88  # reuse
    # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf94, (16, 128, 256), (256, 4096, 1), 0), out=buf96)
    buf97 = reinterpret_tensor(buf94, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf94  # reuse
    cpp_fused_clone_28(c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()))
    buf98 = reinterpret_tensor(buf96, (128, 4096), (4096, 1), 0); del buf96  # reuse
    # Source Nodes: [attn_output_26], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg46_1, (4096, 4096), (1, 4096), 0), out=buf98)
    del arg46_1
    buf99 = reinterpret_tensor(buf79, (128, 16384), (16384, 1), 0); del buf79  # reuse
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg48_1, reinterpret_tensor(buf85, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg47_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf99)
    del arg47_1
    del arg48_1
    buf100 = reinterpret_tensor(buf99, (1, 128, 16384), (2097152, 16384, 1), 0); del buf99  # reuse
    cpp_fused_add_mul_pow_tanh_29(c_void_p(buf100.data_ptr()))
    buf101 = reinterpret_tensor(buf85, (128, 4096), (4096, 1), 0); del buf85  # reuse
    # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg50_1, reinterpret_tensor(buf100, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg49_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf101)
    del arg49_1
    del arg50_1
    buf102 = buf83; del buf83  # reuse
    buf103 = buf82; del buf82  # reuse
    buf105 = reinterpret_tensor(buf97, (1, 128, 4096), (524288, 4096, 1), 0); del buf97  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf98.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg51_1
    del arg52_1
    buf106 = buf87; del buf87  # reuse
    # Source Nodes: [query_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg53_1, (4096, 4096), (1, 4096), 0), out=buf106)
    del arg53_1
    buf107 = buf86; del buf86  # reuse
    # Source Nodes: [key_25], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg54_1, (4096, 4096), (1, 4096), 0), out=buf107)
    del arg54_1
    buf108 = reinterpret_tensor(buf66, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf66  # reuse
    buf109 = reinterpret_tensor(buf65, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf65  # reuse
    cpp_fused_cat_31(c_void_p(buf106.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg300_1
    buf110 = reinterpret_tensor(buf95, (16, 128, 128), (16384, 128, 1), 0); del buf95  # reuse
    # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf109, (16, 256, 128), (256, 1, 4096), 0), out=buf110)
    buf111 = buf93; del buf93  # reuse
    buf112 = reinterpret_tensor(buf110, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf110  # reuse
    buf113 = buf91; del buf91  # reuse
    cpp_fused__softmax_div_lift_fresh_where_32(c_void_p(buf112.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()))
    del arg301_1
    del arg302_1
    buf114 = reinterpret_tensor(buf109, (128, 4096), (4096, 1), 0); del buf109  # reuse
    # Source Nodes: [value_10], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf105, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg55_1, (4096, 4096), (1, 4096), 0), out=buf114)
    del arg55_1
    buf115 = buf112; del buf112  # reuse
    cpp_fused__softmax_33(c_void_p(buf115.data_ptr()), c_void_p(buf113.data_ptr()))
    buf116 = reinterpret_tensor(buf108, (16, 128, 256), (32768, 256, 1), 0); del buf108  # reuse
    # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf115, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf114, (16, 128, 256), (256, 4096, 1), 0), out=buf116)
    buf117 = reinterpret_tensor(buf114, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf114  # reuse
    cpp_fused_clone_34(c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    buf118 = reinterpret_tensor(buf116, (128, 4096), (4096, 1), 0); del buf116  # reuse
    # Source Nodes: [attn_output_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf117, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg56_1, (4096, 4096), (1, 4096), 0), out=buf118)
    del arg56_1
    buf119 = reinterpret_tensor(buf100, (128, 16384), (16384, 1), 0); del buf100  # reuse
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg58_1, reinterpret_tensor(buf105, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg57_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf119)
    del arg57_1
    del arg58_1
    buf120 = reinterpret_tensor(buf119, (1, 128, 16384), (2097152, 16384, 1), 0); del buf119  # reuse
    cpp_fused_add_mul_pow_tanh_35(c_void_p(buf120.data_ptr()))
    buf121 = reinterpret_tensor(buf105, (128, 4096), (4096, 1), 0); del buf105  # reuse
    # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg60_1, reinterpret_tensor(buf120, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg59_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf121)
    del arg59_1
    del arg60_1
    buf122 = reinterpret_tensor(buf118, (1, 128, 4096), (524288, 4096, 1), 0); del buf118  # reuse
    buf123 = buf103; del buf103  # reuse
    buf124 = buf102; del buf102  # reuse
    buf126 = reinterpret_tensor(buf117, (1, 128, 4096), (524288, 4096, 1), 0); del buf117  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf122.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()))
    del arg61_1
    del arg62_1
    buf127 = buf98; del buf98  # reuse
    # Source Nodes: [query_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg63_1, (4096, 4096), (1, 4096), 0), out=buf127)
    del arg63_1
    buf128 = reinterpret_tensor(buf81, (128, 4096), (4096, 1), 0); del buf81  # reuse
    # Source Nodes: [key_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg64_1, (4096, 4096), (1, 4096), 0), out=buf128)
    del arg64_1
    buf129 = reinterpret_tensor(buf121, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf121  # reuse
    buf130 = reinterpret_tensor(buf101, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf101  # reuse
    cpp_fused_cat_37(c_void_p(buf127.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg303_1
    buf131 = reinterpret_tensor(buf115, (16, 128, 128), (16384, 128, 1), 0); del buf115  # reuse
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf129, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf130, (16, 256, 128), (256, 1, 4096), 0), out=buf131)
    buf132 = buf113; del buf113  # reuse
    buf133 = reinterpret_tensor(buf131, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf131  # reuse
    buf134 = buf111; del buf111  # reuse
    cpp_fused__softmax_div_lift_fresh_where_38(c_void_p(buf133.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()))
    del arg304_1
    del arg305_1
    buf135 = reinterpret_tensor(buf130, (128, 4096), (4096, 1), 0); del buf130  # reuse
    # Source Nodes: [value_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf126, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg65_1, (4096, 4096), (1, 4096), 0), out=buf135)
    del arg65_1
    buf136 = buf133; del buf133  # reuse
    cpp_fused__softmax_39(c_void_p(buf136.data_ptr()), c_void_p(buf134.data_ptr()))
    buf137 = reinterpret_tensor(buf129, (16, 128, 256), (32768, 256, 1), 0); del buf129  # reuse
    # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf136, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf135, (16, 128, 256), (256, 4096, 1), 0), out=buf137)
    buf138 = reinterpret_tensor(buf135, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf135  # reuse
    cpp_fused_clone_40(c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()))
    buf139 = reinterpret_tensor(buf137, (128, 4096), (4096, 1), 0); del buf137  # reuse
    # Source Nodes: [attn_output_38], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg66_1, (4096, 4096), (1, 4096), 0), out=buf139)
    del arg66_1
    buf140 = reinterpret_tensor(buf120, (128, 16384), (16384, 1), 0); del buf120  # reuse
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg68_1, reinterpret_tensor(buf126, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg67_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf140)
    del arg67_1
    del arg68_1
    buf141 = reinterpret_tensor(buf140, (1, 128, 16384), (2097152, 16384, 1), 0); del buf140  # reuse
    cpp_fused_add_mul_pow_tanh_41(c_void_p(buf141.data_ptr()))
    buf142 = reinterpret_tensor(buf126, (128, 4096), (4096, 1), 0); del buf126  # reuse
    # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg70_1, reinterpret_tensor(buf141, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg69_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf142)
    del arg69_1
    del arg70_1
    buf143 = buf124; del buf124  # reuse
    buf144 = buf123; del buf123  # reuse
    buf146 = reinterpret_tensor(buf138, (1, 128, 4096), (524288, 4096, 1), 0); del buf138  # reuse
    cpp_fused_add_native_layer_norm_42(c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()))
    del arg71_1
    del arg72_1
    buf147 = buf128; del buf128  # reuse
    # Source Nodes: [query_35], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg73_1, (4096, 4096), (1, 4096), 0), out=buf147)
    del arg73_1
    buf148 = buf127; del buf127  # reuse
    # Source Nodes: [key_35], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg74_1, (4096, 4096), (1, 4096), 0), out=buf148)
    del arg74_1
    buf149 = reinterpret_tensor(buf107, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf107  # reuse
    buf150 = reinterpret_tensor(buf106, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf106  # reuse
    cpp_fused_cat_43(c_void_p(buf147.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    del arg306_1
    buf151 = reinterpret_tensor(buf136, (16, 128, 128), (16384, 128, 1), 0); del buf136  # reuse
    # Source Nodes: [attn_weights_49], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf150, (16, 256, 128), (256, 1, 4096), 0), out=buf151)
    buf152 = buf134; del buf134  # reuse
    buf153 = reinterpret_tensor(buf151, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf151  # reuse
    buf154 = buf132; del buf132  # reuse
    cpp_fused__softmax_div_lift_fresh_where_44(c_void_p(buf153.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()))
    del arg307_1
    del arg308_1
    buf155 = reinterpret_tensor(buf150, (128, 4096), (4096, 1), 0); del buf150  # reuse
    # Source Nodes: [value_14], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf146, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg75_1, (4096, 4096), (1, 4096), 0), out=buf155)
    del arg75_1
    buf156 = buf153; del buf153  # reuse
    cpp_fused__softmax_45(c_void_p(buf156.data_ptr()), c_void_p(buf154.data_ptr()))
    buf157 = reinterpret_tensor(buf149, (16, 128, 256), (32768, 256, 1), 0); del buf149  # reuse
    # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf156, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf155, (16, 128, 256), (256, 4096, 1), 0), out=buf157)
    buf158 = reinterpret_tensor(buf155, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf155  # reuse
    cpp_fused_clone_46(c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf157, (128, 4096), (4096, 1), 0); del buf157  # reuse
    # Source Nodes: [attn_output_44], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf158, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg76_1, (4096, 4096), (1, 4096), 0), out=buf159)
    del arg76_1
    buf160 = reinterpret_tensor(buf141, (128, 16384), (16384, 1), 0); del buf141  # reuse
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg78_1, reinterpret_tensor(buf146, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg77_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf160)
    del arg77_1
    del arg78_1
    buf161 = reinterpret_tensor(buf160, (1, 128, 16384), (2097152, 16384, 1), 0); del buf160  # reuse
    cpp_fused_add_mul_pow_tanh_47(c_void_p(buf161.data_ptr()))
    buf162 = reinterpret_tensor(buf146, (128, 4096), (4096, 1), 0); del buf146  # reuse
    # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg80_1, reinterpret_tensor(buf161, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg79_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf162)
    del arg79_1
    del arg80_1
    buf163 = reinterpret_tensor(buf159, (1, 128, 4096), (524288, 4096, 1), 0); del buf159  # reuse
    buf164 = buf144; del buf144  # reuse
    buf165 = buf143; del buf143  # reuse
    buf167 = reinterpret_tensor(buf158, (1, 128, 4096), (524288, 4096, 1), 0); del buf158  # reuse
    cpp_fused_add_native_layer_norm_48(c_void_p(buf163.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()))
    del arg81_1
    del arg82_1
    buf168 = buf162; del buf162  # reuse
    # Source Nodes: [query_40], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg83_1, (4096, 4096), (1, 4096), 0), out=buf168)
    del arg83_1
    buf169 = buf142; del buf142  # reuse
    # Source Nodes: [key_40], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg84_1, (4096, 4096), (1, 4096), 0), out=buf169)
    del arg84_1
    buf170 = reinterpret_tensor(buf139, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf139  # reuse
    buf171 = reinterpret_tensor(buf122, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf122  # reuse
    cpp_fused_cat_49(c_void_p(buf168.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    del arg309_1
    buf172 = reinterpret_tensor(buf156, (16, 128, 128), (16384, 128, 1), 0); del buf156  # reuse
    # Source Nodes: [attn_weights_56], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf170, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf171, (16, 256, 128), (256, 1, 4096), 0), out=buf172)
    buf173 = buf154; del buf154  # reuse
    buf174 = reinterpret_tensor(buf172, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf172  # reuse
    buf175 = buf152; del buf152  # reuse
    cpp_fused__softmax_div_lift_fresh_where_50(c_void_p(buf174.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()))
    del arg310_1
    del arg311_1
    buf176 = reinterpret_tensor(buf171, (128, 4096), (4096, 1), 0); del buf171  # reuse
    # Source Nodes: [value_16], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf167, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg85_1, (4096, 4096), (1, 4096), 0), out=buf176)
    del arg85_1
    buf177 = buf174; del buf174  # reuse
    cpp_fused__softmax_51(c_void_p(buf177.data_ptr()), c_void_p(buf175.data_ptr()))
    buf178 = reinterpret_tensor(buf170, (16, 128, 256), (32768, 256, 1), 0); del buf170  # reuse
    # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf177, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf176, (16, 128, 256), (256, 4096, 1), 0), out=buf178)
    buf179 = reinterpret_tensor(buf176, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf176  # reuse
    cpp_fused_clone_52(c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    buf180 = reinterpret_tensor(buf178, (128, 4096), (4096, 1), 0); del buf178  # reuse
    # Source Nodes: [attn_output_50], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf179, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg86_1, (4096, 4096), (1, 4096), 0), out=buf180)
    del arg86_1
    buf181 = reinterpret_tensor(buf161, (128, 16384), (16384, 1), 0); del buf161  # reuse
    # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf167, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg87_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf181)
    del arg87_1
    del arg88_1
    buf182 = reinterpret_tensor(buf181, (1, 128, 16384), (2097152, 16384, 1), 0); del buf181  # reuse
    cpp_fused_add_mul_pow_tanh_53(c_void_p(buf182.data_ptr()))
    buf183 = reinterpret_tensor(buf167, (128, 4096), (4096, 1), 0); del buf167  # reuse
    # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg90_1, reinterpret_tensor(buf182, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg89_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf183)
    del arg89_1
    del arg90_1
    buf184 = buf165; del buf165  # reuse
    buf185 = buf164; del buf164  # reuse
    buf187 = reinterpret_tensor(buf179, (1, 128, 4096), (524288, 4096, 1), 0); del buf179  # reuse
    cpp_fused_add_native_layer_norm_54(c_void_p(buf180.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg91_1
    del arg92_1
    buf188 = buf169; del buf169  # reuse
    # Source Nodes: [query_45], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg93_1, (4096, 4096), (1, 4096), 0), out=buf188)
    del arg93_1
    buf189 = buf168; del buf168  # reuse
    # Source Nodes: [key_45], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg94_1, (4096, 4096), (1, 4096), 0), out=buf189)
    del arg94_1
    buf190 = reinterpret_tensor(buf148, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf148  # reuse
    buf191 = reinterpret_tensor(buf147, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf147  # reuse
    cpp_fused_cat_55(c_void_p(buf188.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    del arg312_1
    buf192 = reinterpret_tensor(buf177, (16, 128, 128), (16384, 128, 1), 0); del buf177  # reuse
    # Source Nodes: [attn_weights_63], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf191, (16, 256, 128), (256, 1, 4096), 0), out=buf192)
    buf193 = buf175; del buf175  # reuse
    buf194 = reinterpret_tensor(buf192, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf192  # reuse
    buf195 = buf173; del buf173  # reuse
    cpp_fused__softmax_div_lift_fresh_where_56(c_void_p(buf194.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del arg313_1
    del arg314_1
    buf196 = reinterpret_tensor(buf191, (128, 4096), (4096, 1), 0); del buf191  # reuse
    # Source Nodes: [value_18], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg95_1, (4096, 4096), (1, 4096), 0), out=buf196)
    del arg95_1
    buf197 = buf194; del buf194  # reuse
    cpp_fused__softmax_57(c_void_p(buf197.data_ptr()), c_void_p(buf195.data_ptr()))
    buf198 = reinterpret_tensor(buf190, (16, 128, 256), (32768, 256, 1), 0); del buf190  # reuse
    # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf197, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf196, (16, 128, 256), (256, 4096, 1), 0), out=buf198)
    buf199 = reinterpret_tensor(buf196, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf196  # reuse
    cpp_fused_clone_58(c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf198, (128, 4096), (4096, 1), 0); del buf198  # reuse
    # Source Nodes: [attn_output_56], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg96_1, (4096, 4096), (1, 4096), 0), out=buf200)
    del arg96_1
    buf201 = reinterpret_tensor(buf182, (128, 16384), (16384, 1), 0); del buf182  # reuse
    # Source Nodes: [hidden_states_55], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg98_1, reinterpret_tensor(buf187, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg97_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf201)
    del arg97_1
    del arg98_1
    buf202 = reinterpret_tensor(buf201, (1, 128, 16384), (2097152, 16384, 1), 0); del buf201  # reuse
    cpp_fused_add_mul_pow_tanh_59(c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf187, (128, 4096), (4096, 1), 0); del buf187  # reuse
    # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg100_1, reinterpret_tensor(buf202, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg99_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf203)
    del arg100_1
    del arg99_1
    buf204 = reinterpret_tensor(buf200, (1, 128, 4096), (524288, 4096, 1), 0); del buf200  # reuse
    buf205 = buf185; del buf185  # reuse
    buf206 = buf184; del buf184  # reuse
    buf208 = reinterpret_tensor(buf199, (1, 128, 4096), (524288, 4096, 1), 0); del buf199  # reuse
    cpp_fused_add_native_layer_norm_60(c_void_p(buf204.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()))
    del arg101_1
    del arg102_1
    buf209 = buf203; del buf203  # reuse
    # Source Nodes: [query_50], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg103_1, (4096, 4096), (1, 4096), 0), out=buf209)
    del arg103_1
    buf210 = buf183; del buf183  # reuse
    # Source Nodes: [key_50], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg104_1, (4096, 4096), (1, 4096), 0), out=buf210)
    del arg104_1
    buf211 = reinterpret_tensor(buf180, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf180  # reuse
    buf212 = reinterpret_tensor(buf163, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf163  # reuse
    cpp_fused_cat_61(c_void_p(buf209.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    del arg315_1
    buf213 = reinterpret_tensor(buf197, (16, 128, 128), (16384, 128, 1), 0); del buf197  # reuse
    # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf211, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf212, (16, 256, 128), (256, 1, 4096), 0), out=buf213)
    buf214 = buf195; del buf195  # reuse
    buf215 = reinterpret_tensor(buf213, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf213  # reuse
    buf216 = buf193; del buf193  # reuse
    cpp_fused__softmax_div_lift_fresh_where_62(c_void_p(buf215.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg316_1
    del arg317_1
    buf217 = reinterpret_tensor(buf212, (128, 4096), (4096, 1), 0); del buf212  # reuse
    # Source Nodes: [value_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg105_1, (4096, 4096), (1, 4096), 0), out=buf217)
    del arg105_1
    buf218 = buf215; del buf215  # reuse
    cpp_fused__softmax_63(c_void_p(buf218.data_ptr()), c_void_p(buf216.data_ptr()))
    buf219 = reinterpret_tensor(buf211, (16, 128, 256), (32768, 256, 1), 0); del buf211  # reuse
    # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf218, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf217, (16, 128, 256), (256, 4096, 1), 0), out=buf219)
    buf220 = reinterpret_tensor(buf217, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf217  # reuse
    cpp_fused_clone_64(c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    buf221 = reinterpret_tensor(buf219, (128, 4096), (4096, 1), 0); del buf219  # reuse
    # Source Nodes: [attn_output_62], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf220, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg106_1, (4096, 4096), (1, 4096), 0), out=buf221)
    del arg106_1
    buf222 = reinterpret_tensor(buf202, (128, 16384), (16384, 1), 0); del buf202  # reuse
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf208, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg107_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf222)
    del arg107_1
    del arg108_1
    buf223 = reinterpret_tensor(buf222, (1, 128, 16384), (2097152, 16384, 1), 0); del buf222  # reuse
    cpp_fused_add_mul_pow_tanh_65(c_void_p(buf223.data_ptr()))
    buf224 = reinterpret_tensor(buf208, (128, 4096), (4096, 1), 0); del buf208  # reuse
    # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf223, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg109_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf224)
    del arg109_1
    del arg110_1
    buf225 = buf206; del buf206  # reuse
    buf226 = buf205; del buf205  # reuse
    buf228 = reinterpret_tensor(buf220, (1, 128, 4096), (524288, 4096, 1), 0); del buf220  # reuse
    cpp_fused_add_native_layer_norm_66(c_void_p(buf221.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg111_1
    del arg112_1
    buf229 = buf210; del buf210  # reuse
    # Source Nodes: [query_55], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg113_1, (4096, 4096), (1, 4096), 0), out=buf229)
    del arg113_1
    buf230 = buf209; del buf209  # reuse
    # Source Nodes: [key_55], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg114_1, (4096, 4096), (1, 4096), 0), out=buf230)
    del arg114_1
    buf231 = reinterpret_tensor(buf189, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf189  # reuse
    buf232 = reinterpret_tensor(buf188, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf188  # reuse
    cpp_fused_cat_67(c_void_p(buf229.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()))
    del arg318_1
    buf233 = reinterpret_tensor(buf218, (16, 128, 128), (16384, 128, 1), 0); del buf218  # reuse
    # Source Nodes: [attn_weights_77], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf231, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf232, (16, 256, 128), (256, 1, 4096), 0), out=buf233)
    buf234 = buf216; del buf216  # reuse
    buf235 = reinterpret_tensor(buf233, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf233  # reuse
    buf236 = buf214; del buf214  # reuse
    cpp_fused__softmax_div_lift_fresh_where_68(c_void_p(buf235.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf236.data_ptr()))
    del arg319_1
    del arg320_1
    buf237 = reinterpret_tensor(buf232, (128, 4096), (4096, 1), 0); del buf232  # reuse
    # Source Nodes: [value_22], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf228, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg115_1, (4096, 4096), (1, 4096), 0), out=buf237)
    del arg115_1
    buf238 = buf235; del buf235  # reuse
    cpp_fused__softmax_69(c_void_p(buf238.data_ptr()), c_void_p(buf236.data_ptr()))
    buf239 = reinterpret_tensor(buf231, (16, 128, 256), (32768, 256, 1), 0); del buf231  # reuse
    # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf238, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf237, (16, 128, 256), (256, 4096, 1), 0), out=buf239)
    buf240 = reinterpret_tensor(buf237, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf237  # reuse
    cpp_fused_clone_70(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf239, (128, 4096), (4096, 1), 0); del buf239  # reuse
    # Source Nodes: [attn_output_68], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf240, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg116_1, (4096, 4096), (1, 4096), 0), out=buf241)
    del arg116_1
    buf242 = reinterpret_tensor(buf223, (128, 16384), (16384, 1), 0); del buf223  # reuse
    # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg118_1, reinterpret_tensor(buf228, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg117_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf242)
    del arg117_1
    del arg118_1
    buf243 = reinterpret_tensor(buf242, (1, 128, 16384), (2097152, 16384, 1), 0); del buf242  # reuse
    cpp_fused_add_mul_pow_tanh_71(c_void_p(buf243.data_ptr()))
    buf244 = reinterpret_tensor(buf228, (128, 4096), (4096, 1), 0); del buf228  # reuse
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg120_1, reinterpret_tensor(buf243, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg119_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf244)
    del arg119_1
    del arg120_1
    buf245 = reinterpret_tensor(buf241, (1, 128, 4096), (524288, 4096, 1), 0); del buf241  # reuse
    buf246 = buf226; del buf226  # reuse
    buf247 = buf225; del buf225  # reuse
    buf249 = reinterpret_tensor(buf240, (1, 128, 4096), (524288, 4096, 1), 0); del buf240  # reuse
    cpp_fused_add_native_layer_norm_72(c_void_p(buf245.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf249.data_ptr()))
    del arg121_1
    del arg122_1
    buf250 = buf244; del buf244  # reuse
    # Source Nodes: [query_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg123_1, (4096, 4096), (1, 4096), 0), out=buf250)
    del arg123_1
    buf251 = buf224; del buf224  # reuse
    # Source Nodes: [key_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg124_1, (4096, 4096), (1, 4096), 0), out=buf251)
    del arg124_1
    buf252 = reinterpret_tensor(buf221, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf221  # reuse
    buf253 = reinterpret_tensor(buf204, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf204  # reuse
    cpp_fused_cat_73(c_void_p(buf250.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    del arg321_1
    buf254 = reinterpret_tensor(buf238, (16, 128, 128), (16384, 128, 1), 0); del buf238  # reuse
    # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf252, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf253, (16, 256, 128), (256, 1, 4096), 0), out=buf254)
    buf255 = buf236; del buf236  # reuse
    buf256 = reinterpret_tensor(buf254, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf254  # reuse
    buf257 = buf234; del buf234  # reuse
    cpp_fused__softmax_div_lift_fresh_where_74(c_void_p(buf256.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del arg322_1
    del arg323_1
    buf258 = reinterpret_tensor(buf253, (128, 4096), (4096, 1), 0); del buf253  # reuse
    # Source Nodes: [value_24], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg125_1, (4096, 4096), (1, 4096), 0), out=buf258)
    del arg125_1
    buf259 = buf256; del buf256  # reuse
    cpp_fused__softmax_75(c_void_p(buf259.data_ptr()), c_void_p(buf257.data_ptr()))
    buf260 = reinterpret_tensor(buf252, (16, 128, 256), (32768, 256, 1), 0); del buf252  # reuse
    # Source Nodes: [attn_output_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf258, (16, 128, 256), (256, 4096, 1), 0), out=buf260)
    buf261 = reinterpret_tensor(buf258, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf258  # reuse
    cpp_fused_clone_76(c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    buf262 = reinterpret_tensor(buf260, (128, 4096), (4096, 1), 0); del buf260  # reuse
    # Source Nodes: [attn_output_74], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf261, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg126_1, (4096, 4096), (1, 4096), 0), out=buf262)
    del arg126_1
    buf263 = reinterpret_tensor(buf243, (128, 16384), (16384, 1), 0); del buf243  # reuse
    # Source Nodes: [hidden_states_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg128_1, reinterpret_tensor(buf249, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg127_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf263)
    del arg127_1
    del arg128_1
    buf264 = reinterpret_tensor(buf263, (1, 128, 16384), (2097152, 16384, 1), 0); del buf263  # reuse
    cpp_fused_add_mul_pow_tanh_77(c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf249, (128, 4096), (4096, 1), 0); del buf249  # reuse
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf264, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg129_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf265)
    del arg129_1
    del arg130_1
    buf266 = buf247; del buf247  # reuse
    buf267 = buf246; del buf246  # reuse
    buf269 = reinterpret_tensor(buf261, (1, 128, 4096), (524288, 4096, 1), 0); del buf261  # reuse
    cpp_fused_add_native_layer_norm_78(c_void_p(buf262.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()))
    del arg131_1
    del arg132_1
    buf270 = buf251; del buf251  # reuse
    # Source Nodes: [query_65], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg133_1, (4096, 4096), (1, 4096), 0), out=buf270)
    del arg133_1
    buf271 = buf250; del buf250  # reuse
    # Source Nodes: [key_65], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg134_1, (4096, 4096), (1, 4096), 0), out=buf271)
    del arg134_1
    buf272 = reinterpret_tensor(buf230, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf230  # reuse
    buf273 = reinterpret_tensor(buf229, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf229  # reuse
    cpp_fused_cat_79(c_void_p(buf270.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    del arg324_1
    buf274 = reinterpret_tensor(buf259, (16, 128, 128), (16384, 128, 1), 0); del buf259  # reuse
    # Source Nodes: [attn_weights_91], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf272, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf273, (16, 256, 128), (256, 1, 4096), 0), out=buf274)
    buf275 = buf257; del buf257  # reuse
    buf276 = reinterpret_tensor(buf274, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf274  # reuse
    buf277 = buf255; del buf255  # reuse
    cpp_fused__softmax_div_lift_fresh_where_80(c_void_p(buf276.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()))
    del arg325_1
    del arg326_1
    buf278 = reinterpret_tensor(buf273, (128, 4096), (4096, 1), 0); del buf273  # reuse
    # Source Nodes: [value_26], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf269, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg135_1, (4096, 4096), (1, 4096), 0), out=buf278)
    del arg135_1
    buf279 = buf276; del buf276  # reuse
    cpp_fused__softmax_81(c_void_p(buf279.data_ptr()), c_void_p(buf277.data_ptr()))
    buf280 = reinterpret_tensor(buf272, (16, 128, 256), (32768, 256, 1), 0); del buf272  # reuse
    # Source Nodes: [attn_output_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf279, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf278, (16, 128, 256), (256, 4096, 1), 0), out=buf280)
    buf281 = reinterpret_tensor(buf278, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf278  # reuse
    cpp_fused_clone_82(c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    buf282 = reinterpret_tensor(buf280, (128, 4096), (4096, 1), 0); del buf280  # reuse
    # Source Nodes: [attn_output_80], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf281, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg136_1, (4096, 4096), (1, 4096), 0), out=buf282)
    del arg136_1
    buf283 = reinterpret_tensor(buf264, (128, 16384), (16384, 1), 0); del buf264  # reuse
    # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg138_1, reinterpret_tensor(buf269, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg137_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf283)
    del arg137_1
    del arg138_1
    buf284 = reinterpret_tensor(buf283, (1, 128, 16384), (2097152, 16384, 1), 0); del buf283  # reuse
    cpp_fused_add_mul_pow_tanh_83(c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf269, (128, 4096), (4096, 1), 0); del buf269  # reuse
    # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg140_1, reinterpret_tensor(buf284, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg139_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf285)
    del arg139_1
    del arg140_1
    buf286 = reinterpret_tensor(buf282, (1, 128, 4096), (524288, 4096, 1), 0); del buf282  # reuse
    buf287 = buf267; del buf267  # reuse
    buf288 = buf266; del buf266  # reuse
    buf290 = reinterpret_tensor(buf281, (1, 128, 4096), (524288, 4096, 1), 0); del buf281  # reuse
    cpp_fused_add_native_layer_norm_84(c_void_p(buf286.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()))
    del arg141_1
    del arg142_1
    buf291 = buf285; del buf285  # reuse
    # Source Nodes: [query_70], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf290, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg143_1, (4096, 4096), (1, 4096), 0), out=buf291)
    del arg143_1
    buf292 = buf265; del buf265  # reuse
    # Source Nodes: [key_70], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf290, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg144_1, (4096, 4096), (1, 4096), 0), out=buf292)
    del arg144_1
    buf293 = reinterpret_tensor(buf262, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf262  # reuse
    buf294 = reinterpret_tensor(buf245, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf245  # reuse
    cpp_fused_cat_85(c_void_p(buf291.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    del arg327_1
    buf295 = reinterpret_tensor(buf279, (16, 128, 128), (16384, 128, 1), 0); del buf279  # reuse
    # Source Nodes: [attn_weights_98], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf293, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf294, (16, 256, 128), (256, 1, 4096), 0), out=buf295)
    buf296 = buf277; del buf277  # reuse
    buf297 = reinterpret_tensor(buf295, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf295  # reuse
    buf298 = buf275; del buf275  # reuse
    cpp_fused__softmax_div_lift_fresh_where_86(c_void_p(buf297.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del arg328_1
    del arg329_1
    buf299 = reinterpret_tensor(buf294, (128, 4096), (4096, 1), 0); del buf294  # reuse
    # Source Nodes: [value_28], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf290, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg145_1, (4096, 4096), (1, 4096), 0), out=buf299)
    del arg145_1
    buf300 = buf297; del buf297  # reuse
    cpp_fused__softmax_87(c_void_p(buf300.data_ptr()), c_void_p(buf298.data_ptr()))
    buf301 = reinterpret_tensor(buf293, (16, 128, 256), (32768, 256, 1), 0); del buf293  # reuse
    # Source Nodes: [attn_output_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf300, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf299, (16, 128, 256), (256, 4096, 1), 0), out=buf301)
    buf302 = reinterpret_tensor(buf299, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf299  # reuse
    cpp_fused_clone_88(c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf301, (128, 4096), (4096, 1), 0); del buf301  # reuse
    # Source Nodes: [attn_output_86], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf302, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg146_1, (4096, 4096), (1, 4096), 0), out=buf303)
    del arg146_1
    buf304 = reinterpret_tensor(buf284, (128, 16384), (16384, 1), 0); del buf284  # reuse
    # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf290, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg147_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf304)
    del arg147_1
    del arg148_1
    buf305 = reinterpret_tensor(buf304, (1, 128, 16384), (2097152, 16384, 1), 0); del buf304  # reuse
    cpp_fused_add_mul_pow_tanh_89(c_void_p(buf305.data_ptr()))
    buf306 = reinterpret_tensor(buf290, (128, 4096), (4096, 1), 0); del buf290  # reuse
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg150_1, reinterpret_tensor(buf305, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg149_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf306)
    del arg149_1
    del arg150_1
    buf307 = buf288; del buf288  # reuse
    buf308 = buf287; del buf287  # reuse
    buf310 = reinterpret_tensor(buf302, (1, 128, 4096), (524288, 4096, 1), 0); del buf302  # reuse
    cpp_fused_add_native_layer_norm_90(c_void_p(buf303.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()))
    del arg151_1
    del arg152_1
    buf311 = buf292; del buf292  # reuse
    # Source Nodes: [query_75], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg153_1, (4096, 4096), (1, 4096), 0), out=buf311)
    del arg153_1
    buf312 = buf291; del buf291  # reuse
    # Source Nodes: [key_75], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg154_1, (4096, 4096), (1, 4096), 0), out=buf312)
    del arg154_1
    buf313 = reinterpret_tensor(buf271, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf271  # reuse
    buf314 = reinterpret_tensor(buf270, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf270  # reuse
    cpp_fused_cat_91(c_void_p(buf311.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    del arg330_1
    buf315 = reinterpret_tensor(buf300, (16, 128, 128), (16384, 128, 1), 0); del buf300  # reuse
    # Source Nodes: [attn_weights_105], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf313, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf314, (16, 256, 128), (256, 1, 4096), 0), out=buf315)
    buf316 = buf298; del buf298  # reuse
    buf317 = reinterpret_tensor(buf315, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf315  # reuse
    buf318 = buf296; del buf296  # reuse
    cpp_fused__softmax_div_lift_fresh_where_92(c_void_p(buf317.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()))
    del arg331_1
    del arg332_1
    buf319 = reinterpret_tensor(buf314, (128, 4096), (4096, 1), 0); del buf314  # reuse
    # Source Nodes: [value_30], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf310, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg155_1, (4096, 4096), (1, 4096), 0), out=buf319)
    del arg155_1
    buf320 = buf317; del buf317  # reuse
    cpp_fused__softmax_93(c_void_p(buf320.data_ptr()), c_void_p(buf318.data_ptr()))
    buf321 = reinterpret_tensor(buf313, (16, 128, 256), (32768, 256, 1), 0); del buf313  # reuse
    # Source Nodes: [attn_output_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf320, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf319, (16, 128, 256), (256, 4096, 1), 0), out=buf321)
    buf322 = reinterpret_tensor(buf319, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf319  # reuse
    cpp_fused_clone_94(c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()))
    buf323 = reinterpret_tensor(buf321, (128, 4096), (4096, 1), 0); del buf321  # reuse
    # Source Nodes: [attn_output_92], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf322, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg156_1, (4096, 4096), (1, 4096), 0), out=buf323)
    del arg156_1
    buf324 = reinterpret_tensor(buf305, (128, 16384), (16384, 1), 0); del buf305  # reuse
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg158_1, reinterpret_tensor(buf310, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg157_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf324)
    del arg157_1
    del arg158_1
    buf325 = reinterpret_tensor(buf324, (1, 128, 16384), (2097152, 16384, 1), 0); del buf324  # reuse
    cpp_fused_add_mul_pow_tanh_95(c_void_p(buf325.data_ptr()))
    buf326 = reinterpret_tensor(buf310, (128, 4096), (4096, 1), 0); del buf310  # reuse
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg160_1, reinterpret_tensor(buf325, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg159_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf326)
    del arg159_1
    del arg160_1
    buf327 = reinterpret_tensor(buf323, (1, 128, 4096), (524288, 4096, 1), 0); del buf323  # reuse
    buf328 = buf308; del buf308  # reuse
    buf329 = buf307; del buf307  # reuse
    buf331 = reinterpret_tensor(buf322, (1, 128, 4096), (524288, 4096, 1), 0); del buf322  # reuse
    cpp_fused_add_native_layer_norm_96(c_void_p(buf327.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()))
    del arg161_1
    del arg162_1
    buf332 = buf326; del buf326  # reuse
    # Source Nodes: [query_80], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg163_1, (4096, 4096), (1, 4096), 0), out=buf332)
    del arg163_1
    buf333 = buf306; del buf306  # reuse
    # Source Nodes: [key_80], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg164_1, (4096, 4096), (1, 4096), 0), out=buf333)
    del arg164_1
    buf334 = reinterpret_tensor(buf303, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf303  # reuse
    buf335 = reinterpret_tensor(buf286, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf286  # reuse
    cpp_fused_cat_97(c_void_p(buf332.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()))
    del arg333_1
    buf336 = reinterpret_tensor(buf320, (16, 128, 128), (16384, 128, 1), 0); del buf320  # reuse
    # Source Nodes: [attn_weights_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf334, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf335, (16, 256, 128), (256, 1, 4096), 0), out=buf336)
    buf337 = buf318; del buf318  # reuse
    buf338 = reinterpret_tensor(buf336, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf336  # reuse
    buf339 = buf316; del buf316  # reuse
    cpp_fused__softmax_div_lift_fresh_where_98(c_void_p(buf338.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()))
    del arg334_1
    del arg335_1
    buf340 = reinterpret_tensor(buf335, (128, 4096), (4096, 1), 0); del buf335  # reuse
    # Source Nodes: [value_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg165_1, (4096, 4096), (1, 4096), 0), out=buf340)
    del arg165_1
    buf341 = buf338; del buf338  # reuse
    cpp_fused__softmax_99(c_void_p(buf341.data_ptr()), c_void_p(buf339.data_ptr()))
    buf342 = reinterpret_tensor(buf334, (16, 128, 256), (32768, 256, 1), 0); del buf334  # reuse
    # Source Nodes: [attn_output_96], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf341, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf340, (16, 128, 256), (256, 4096, 1), 0), out=buf342)
    buf343 = reinterpret_tensor(buf340, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf340  # reuse
    cpp_fused_clone_100(c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()))
    buf344 = reinterpret_tensor(buf342, (128, 4096), (4096, 1), 0); del buf342  # reuse
    # Source Nodes: [attn_output_98], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf343, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg166_1, (4096, 4096), (1, 4096), 0), out=buf344)
    del arg166_1
    buf345 = reinterpret_tensor(buf325, (128, 16384), (16384, 1), 0); del buf325  # reuse
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf331, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg167_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf345)
    del arg167_1
    del arg168_1
    buf346 = reinterpret_tensor(buf345, (1, 128, 16384), (2097152, 16384, 1), 0); del buf345  # reuse
    cpp_fused_add_mul_pow_tanh_101(c_void_p(buf346.data_ptr()))
    buf347 = reinterpret_tensor(buf331, (128, 4096), (4096, 1), 0); del buf331  # reuse
    # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf346, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg169_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf347)
    del arg169_1
    del arg170_1
    buf348 = buf329; del buf329  # reuse
    buf349 = buf328; del buf328  # reuse
    buf351 = reinterpret_tensor(buf343, (1, 128, 4096), (524288, 4096, 1), 0); del buf343  # reuse
    cpp_fused_add_native_layer_norm_102(c_void_p(buf344.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()))
    del arg171_1
    del arg172_1
    buf352 = buf333; del buf333  # reuse
    # Source Nodes: [query_85], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg173_1, (4096, 4096), (1, 4096), 0), out=buf352)
    del arg173_1
    buf353 = buf332; del buf332  # reuse
    # Source Nodes: [key_85], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg174_1, (4096, 4096), (1, 4096), 0), out=buf353)
    del arg174_1
    buf354 = reinterpret_tensor(buf312, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf312  # reuse
    buf355 = reinterpret_tensor(buf311, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf311  # reuse
    cpp_fused_cat_103(c_void_p(buf352.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del arg336_1
    buf356 = reinterpret_tensor(buf341, (16, 128, 128), (16384, 128, 1), 0); del buf341  # reuse
    # Source Nodes: [attn_weights_119], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf354, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf355, (16, 256, 128), (256, 1, 4096), 0), out=buf356)
    buf357 = buf339; del buf339  # reuse
    buf358 = reinterpret_tensor(buf356, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf356  # reuse
    buf359 = buf337; del buf337  # reuse
    cpp_fused__softmax_div_lift_fresh_where_104(c_void_p(buf358.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf359.data_ptr()))
    del arg337_1
    del arg338_1
    buf360 = reinterpret_tensor(buf355, (128, 4096), (4096, 1), 0); del buf355  # reuse
    # Source Nodes: [value_34], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg175_1, (4096, 4096), (1, 4096), 0), out=buf360)
    del arg175_1
    buf361 = buf358; del buf358  # reuse
    cpp_fused__softmax_105(c_void_p(buf361.data_ptr()), c_void_p(buf359.data_ptr()))
    buf362 = reinterpret_tensor(buf354, (16, 128, 256), (32768, 256, 1), 0); del buf354  # reuse
    # Source Nodes: [attn_output_102], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf361, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf360, (16, 128, 256), (256, 4096, 1), 0), out=buf362)
    buf363 = reinterpret_tensor(buf360, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf360  # reuse
    cpp_fused_clone_106(c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()))
    buf364 = reinterpret_tensor(buf362, (128, 4096), (4096, 1), 0); del buf362  # reuse
    # Source Nodes: [attn_output_104], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf363, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg176_1, (4096, 4096), (1, 4096), 0), out=buf364)
    del arg176_1
    buf365 = reinterpret_tensor(buf346, (128, 16384), (16384, 1), 0); del buf346  # reuse
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg178_1, reinterpret_tensor(buf351, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg177_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf365)
    del arg177_1
    del arg178_1
    buf366 = reinterpret_tensor(buf365, (1, 128, 16384), (2097152, 16384, 1), 0); del buf365  # reuse
    cpp_fused_add_mul_pow_tanh_107(c_void_p(buf366.data_ptr()))
    buf367 = reinterpret_tensor(buf351, (128, 4096), (4096, 1), 0); del buf351  # reuse
    # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg180_1, reinterpret_tensor(buf366, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg179_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf367)
    del arg179_1
    del arg180_1
    buf368 = reinterpret_tensor(buf364, (1, 128, 4096), (524288, 4096, 1), 0); del buf364  # reuse
    buf369 = buf349; del buf349  # reuse
    buf370 = buf348; del buf348  # reuse
    buf372 = reinterpret_tensor(buf363, (1, 128, 4096), (524288, 4096, 1), 0); del buf363  # reuse
    cpp_fused_add_native_layer_norm_108(c_void_p(buf368.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()))
    del arg181_1
    del arg182_1
    buf373 = buf367; del buf367  # reuse
    # Source Nodes: [query_90], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg183_1, (4096, 4096), (1, 4096), 0), out=buf373)
    del arg183_1
    buf374 = buf347; del buf347  # reuse
    # Source Nodes: [key_90], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg184_1, (4096, 4096), (1, 4096), 0), out=buf374)
    del arg184_1
    buf375 = reinterpret_tensor(buf344, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf344  # reuse
    buf376 = reinterpret_tensor(buf327, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf327  # reuse
    cpp_fused_cat_109(c_void_p(buf373.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()))
    del arg339_1
    buf377 = reinterpret_tensor(buf361, (16, 128, 128), (16384, 128, 1), 0); del buf361  # reuse
    # Source Nodes: [attn_weights_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf375, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf376, (16, 256, 128), (256, 1, 4096), 0), out=buf377)
    buf378 = buf359; del buf359  # reuse
    buf379 = reinterpret_tensor(buf377, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf377  # reuse
    buf380 = buf357; del buf357  # reuse
    cpp_fused__softmax_div_lift_fresh_where_110(c_void_p(buf379.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()))
    del arg340_1
    del arg341_1
    buf381 = reinterpret_tensor(buf376, (128, 4096), (4096, 1), 0); del buf376  # reuse
    # Source Nodes: [value_36], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf372, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg185_1, (4096, 4096), (1, 4096), 0), out=buf381)
    del arg185_1
    buf382 = buf379; del buf379  # reuse
    cpp_fused__softmax_111(c_void_p(buf382.data_ptr()), c_void_p(buf380.data_ptr()))
    buf383 = reinterpret_tensor(buf375, (16, 128, 256), (32768, 256, 1), 0); del buf375  # reuse
    # Source Nodes: [attn_output_108], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf382, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf381, (16, 128, 256), (256, 4096, 1), 0), out=buf383)
    buf384 = reinterpret_tensor(buf381, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf381  # reuse
    cpp_fused_clone_112(c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    buf385 = reinterpret_tensor(buf383, (128, 4096), (4096, 1), 0); del buf383  # reuse
    # Source Nodes: [attn_output_110], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf384, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg186_1, (4096, 4096), (1, 4096), 0), out=buf385)
    del arg186_1
    buf386 = reinterpret_tensor(buf366, (128, 16384), (16384, 1), 0); del buf366  # reuse
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg188_1, reinterpret_tensor(buf372, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg187_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf386)
    del arg187_1
    del arg188_1
    buf387 = reinterpret_tensor(buf386, (1, 128, 16384), (2097152, 16384, 1), 0); del buf386  # reuse
    cpp_fused_add_mul_pow_tanh_113(c_void_p(buf387.data_ptr()))
    buf388 = reinterpret_tensor(buf372, (128, 4096), (4096, 1), 0); del buf372  # reuse
    # Source Nodes: [hidden_states_111], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg190_1, reinterpret_tensor(buf387, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg189_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf388)
    del arg189_1
    del arg190_1
    buf389 = buf370; del buf370  # reuse
    buf390 = buf369; del buf369  # reuse
    buf392 = reinterpret_tensor(buf384, (1, 128, 4096), (524288, 4096, 1), 0); del buf384  # reuse
    cpp_fused_add_native_layer_norm_114(c_void_p(buf385.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf392.data_ptr()))
    del arg191_1
    del arg192_1
    buf393 = buf374; del buf374  # reuse
    # Source Nodes: [query_95], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf392, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg193_1, (4096, 4096), (1, 4096), 0), out=buf393)
    del arg193_1
    buf394 = buf373; del buf373  # reuse
    # Source Nodes: [key_95], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf392, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg194_1, (4096, 4096), (1, 4096), 0), out=buf394)
    del arg194_1
    buf395 = reinterpret_tensor(buf353, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf353  # reuse
    buf396 = reinterpret_tensor(buf352, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf352  # reuse
    cpp_fused_cat_115(c_void_p(buf393.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del arg342_1
    buf397 = reinterpret_tensor(buf382, (16, 128, 128), (16384, 128, 1), 0); del buf382  # reuse
    # Source Nodes: [attn_weights_133], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf395, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf396, (16, 256, 128), (256, 1, 4096), 0), out=buf397)
    buf398 = buf380; del buf380  # reuse
    buf399 = reinterpret_tensor(buf397, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf397  # reuse
    buf400 = buf378; del buf378  # reuse
    cpp_fused__softmax_div_lift_fresh_where_116(c_void_p(buf399.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf400.data_ptr()))
    del arg343_1
    del arg344_1
    buf401 = reinterpret_tensor(buf396, (128, 4096), (4096, 1), 0); del buf396  # reuse
    # Source Nodes: [value_38], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf392, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg195_1, (4096, 4096), (1, 4096), 0), out=buf401)
    del arg195_1
    buf402 = buf399; del buf399  # reuse
    cpp_fused__softmax_117(c_void_p(buf402.data_ptr()), c_void_p(buf400.data_ptr()))
    buf403 = reinterpret_tensor(buf395, (16, 128, 256), (32768, 256, 1), 0); del buf395  # reuse
    # Source Nodes: [attn_output_114], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf402, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf401, (16, 128, 256), (256, 4096, 1), 0), out=buf403)
    buf404 = reinterpret_tensor(buf401, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf401  # reuse
    cpp_fused_clone_118(c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    buf405 = reinterpret_tensor(buf403, (128, 4096), (4096, 1), 0); del buf403  # reuse
    # Source Nodes: [attn_output_116], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf404, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg196_1, (4096, 4096), (1, 4096), 0), out=buf405)
    del arg196_1
    buf406 = reinterpret_tensor(buf387, (128, 16384), (16384, 1), 0); del buf387  # reuse
    # Source Nodes: [hidden_states_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf392, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg197_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf406)
    del arg197_1
    del arg198_1
    buf407 = reinterpret_tensor(buf406, (1, 128, 16384), (2097152, 16384, 1), 0); del buf406  # reuse
    cpp_fused_add_mul_pow_tanh_119(c_void_p(buf407.data_ptr()))
    buf408 = reinterpret_tensor(buf392, (128, 4096), (4096, 1), 0); del buf392  # reuse
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg200_1, reinterpret_tensor(buf407, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg199_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf408)
    del arg199_1
    del arg200_1
    buf409 = reinterpret_tensor(buf405, (1, 128, 4096), (524288, 4096, 1), 0); del buf405  # reuse
    buf410 = buf390; del buf390  # reuse
    buf411 = buf389; del buf389  # reuse
    buf413 = reinterpret_tensor(buf404, (1, 128, 4096), (524288, 4096, 1), 0); del buf404  # reuse
    cpp_fused_add_native_layer_norm_120(c_void_p(buf409.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf413.data_ptr()))
    del arg201_1
    del arg202_1
    buf414 = buf408; del buf408  # reuse
    # Source Nodes: [query_100], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg203_1, (4096, 4096), (1, 4096), 0), out=buf414)
    del arg203_1
    buf415 = buf388; del buf388  # reuse
    # Source Nodes: [key_100], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg204_1, (4096, 4096), (1, 4096), 0), out=buf415)
    del arg204_1
    buf416 = reinterpret_tensor(buf385, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf385  # reuse
    buf417 = reinterpret_tensor(buf368, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf368  # reuse
    cpp_fused_cat_121(c_void_p(buf414.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()))
    del arg345_1
    buf418 = reinterpret_tensor(buf402, (16, 128, 128), (16384, 128, 1), 0); del buf402  # reuse
    # Source Nodes: [attn_weights_140], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf416, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf417, (16, 256, 128), (256, 1, 4096), 0), out=buf418)
    buf419 = buf400; del buf400  # reuse
    buf420 = reinterpret_tensor(buf418, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf418  # reuse
    buf421 = buf398; del buf398  # reuse
    cpp_fused__softmax_div_lift_fresh_where_122(c_void_p(buf420.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    del arg346_1
    del arg347_1
    buf422 = reinterpret_tensor(buf417, (128, 4096), (4096, 1), 0); del buf417  # reuse
    # Source Nodes: [value_40], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf413, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg205_1, (4096, 4096), (1, 4096), 0), out=buf422)
    del arg205_1
    buf423 = buf420; del buf420  # reuse
    cpp_fused__softmax_123(c_void_p(buf423.data_ptr()), c_void_p(buf421.data_ptr()))
    buf424 = reinterpret_tensor(buf416, (16, 128, 256), (32768, 256, 1), 0); del buf416  # reuse
    # Source Nodes: [attn_output_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf423, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf422, (16, 128, 256), (256, 4096, 1), 0), out=buf424)
    buf425 = reinterpret_tensor(buf422, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf422  # reuse
    cpp_fused_clone_124(c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()))
    buf426 = reinterpret_tensor(buf424, (128, 4096), (4096, 1), 0); del buf424  # reuse
    # Source Nodes: [attn_output_122], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf425, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg206_1, (4096, 4096), (1, 4096), 0), out=buf426)
    del arg206_1
    buf427 = reinterpret_tensor(buf407, (128, 16384), (16384, 1), 0); del buf407  # reuse
    # Source Nodes: [hidden_states_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg208_1, reinterpret_tensor(buf413, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg207_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf427)
    del arg207_1
    del arg208_1
    buf428 = reinterpret_tensor(buf427, (1, 128, 16384), (2097152, 16384, 1), 0); del buf427  # reuse
    cpp_fused_add_mul_pow_tanh_125(c_void_p(buf428.data_ptr()))
    buf429 = reinterpret_tensor(buf413, (128, 4096), (4096, 1), 0); del buf413  # reuse
    # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg210_1, reinterpret_tensor(buf428, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg209_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf429)
    del arg209_1
    del arg210_1
    buf430 = buf411; del buf411  # reuse
    buf431 = buf410; del buf410  # reuse
    buf433 = reinterpret_tensor(buf425, (1, 128, 4096), (524288, 4096, 1), 0); del buf425  # reuse
    cpp_fused_add_native_layer_norm_126(c_void_p(buf426.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf433.data_ptr()))
    del arg211_1
    del arg212_1
    buf434 = buf415; del buf415  # reuse
    # Source Nodes: [query_105], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf433, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg213_1, (4096, 4096), (1, 4096), 0), out=buf434)
    del arg213_1
    buf435 = buf414; del buf414  # reuse
    # Source Nodes: [key_105], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf433, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg214_1, (4096, 4096), (1, 4096), 0), out=buf435)
    del arg214_1
    buf436 = reinterpret_tensor(buf394, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf394  # reuse
    buf437 = reinterpret_tensor(buf393, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf393  # reuse
    cpp_fused_cat_127(c_void_p(buf434.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    del arg348_1
    buf438 = reinterpret_tensor(buf423, (16, 128, 128), (16384, 128, 1), 0); del buf423  # reuse
    # Source Nodes: [attn_weights_147], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf436, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf437, (16, 256, 128), (256, 1, 4096), 0), out=buf438)
    buf439 = buf421; del buf421  # reuse
    buf440 = reinterpret_tensor(buf438, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf438  # reuse
    buf441 = buf419; del buf419  # reuse
    cpp_fused__softmax_div_lift_fresh_where_128(c_void_p(buf440.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf441.data_ptr()))
    del arg349_1
    del arg350_1
    buf442 = reinterpret_tensor(buf437, (128, 4096), (4096, 1), 0); del buf437  # reuse
    # Source Nodes: [value_42], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf433, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg215_1, (4096, 4096), (1, 4096), 0), out=buf442)
    del arg215_1
    buf443 = buf440; del buf440  # reuse
    cpp_fused__softmax_129(c_void_p(buf443.data_ptr()), c_void_p(buf441.data_ptr()))
    buf444 = reinterpret_tensor(buf436, (16, 128, 256), (32768, 256, 1), 0); del buf436  # reuse
    # Source Nodes: [attn_output_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf443, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf442, (16, 128, 256), (256, 4096, 1), 0), out=buf444)
    buf445 = reinterpret_tensor(buf442, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf442  # reuse
    cpp_fused_clone_130(c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()))
    buf446 = reinterpret_tensor(buf444, (128, 4096), (4096, 1), 0); del buf444  # reuse
    # Source Nodes: [attn_output_128], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf445, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg216_1, (4096, 4096), (1, 4096), 0), out=buf446)
    del arg216_1
    buf447 = reinterpret_tensor(buf428, (128, 16384), (16384, 1), 0); del buf428  # reuse
    # Source Nodes: [hidden_states_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf433, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg217_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf447)
    del arg217_1
    del arg218_1
    buf448 = reinterpret_tensor(buf447, (1, 128, 16384), (2097152, 16384, 1), 0); del buf447  # reuse
    cpp_fused_add_mul_pow_tanh_131(c_void_p(buf448.data_ptr()))
    buf449 = reinterpret_tensor(buf433, (128, 4096), (4096, 1), 0); del buf433  # reuse
    # Source Nodes: [hidden_states_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf448, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg219_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf449)
    del arg219_1
    del arg220_1
    buf450 = reinterpret_tensor(buf446, (1, 128, 4096), (524288, 4096, 1), 0); del buf446  # reuse
    buf451 = buf431; del buf431  # reuse
    buf452 = buf430; del buf430  # reuse
    buf454 = reinterpret_tensor(buf445, (1, 128, 4096), (524288, 4096, 1), 0); del buf445  # reuse
    cpp_fused_add_native_layer_norm_132(c_void_p(buf450.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()))
    del arg221_1
    del arg222_1
    buf455 = buf449; del buf449  # reuse
    # Source Nodes: [query_110], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg223_1, (4096, 4096), (1, 4096), 0), out=buf455)
    del arg223_1
    buf456 = buf429; del buf429  # reuse
    # Source Nodes: [key_110], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg224_1, (4096, 4096), (1, 4096), 0), out=buf456)
    del arg224_1
    buf457 = reinterpret_tensor(buf426, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf426  # reuse
    buf458 = reinterpret_tensor(buf409, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf409  # reuse
    cpp_fused_cat_133(c_void_p(buf455.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()))
    del arg351_1
    buf459 = reinterpret_tensor(buf443, (16, 128, 128), (16384, 128, 1), 0); del buf443  # reuse
    # Source Nodes: [attn_weights_154], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf457, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf458, (16, 256, 128), (256, 1, 4096), 0), out=buf459)
    buf460 = buf441; del buf441  # reuse
    buf461 = reinterpret_tensor(buf459, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf459  # reuse
    buf462 = buf439; del buf439  # reuse
    cpp_fused__softmax_div_lift_fresh_where_134(c_void_p(buf461.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf462.data_ptr()))
    del arg352_1
    del arg353_1
    buf463 = reinterpret_tensor(buf458, (128, 4096), (4096, 1), 0); del buf458  # reuse
    # Source Nodes: [value_44], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf454, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg225_1, (4096, 4096), (1, 4096), 0), out=buf463)
    del arg225_1
    buf464 = buf461; del buf461  # reuse
    cpp_fused__softmax_135(c_void_p(buf464.data_ptr()), c_void_p(buf462.data_ptr()))
    buf465 = reinterpret_tensor(buf457, (16, 128, 256), (32768, 256, 1), 0); del buf457  # reuse
    # Source Nodes: [attn_output_132], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf464, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf463, (16, 128, 256), (256, 4096, 1), 0), out=buf465)
    buf466 = reinterpret_tensor(buf463, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf463  # reuse
    cpp_fused_clone_136(c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    buf467 = reinterpret_tensor(buf465, (128, 4096), (4096, 1), 0); del buf465  # reuse
    # Source Nodes: [attn_output_134], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf466, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg226_1, (4096, 4096), (1, 4096), 0), out=buf467)
    del arg226_1
    buf468 = reinterpret_tensor(buf448, (128, 16384), (16384, 1), 0); del buf448  # reuse
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg228_1, reinterpret_tensor(buf454, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg227_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf468)
    del arg227_1
    del arg228_1
    buf469 = reinterpret_tensor(buf468, (1, 128, 16384), (2097152, 16384, 1), 0); del buf468  # reuse
    cpp_fused_add_mul_pow_tanh_137(c_void_p(buf469.data_ptr()))
    buf470 = reinterpret_tensor(buf454, (128, 4096), (4096, 1), 0); del buf454  # reuse
    # Source Nodes: [hidden_states_135], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg230_1, reinterpret_tensor(buf469, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg229_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf470)
    del arg229_1
    del arg230_1
    buf471 = buf452; del buf452  # reuse
    buf472 = buf451; del buf451  # reuse
    buf474 = reinterpret_tensor(buf466, (1, 128, 4096), (524288, 4096, 1), 0); del buf466  # reuse
    cpp_fused_add_native_layer_norm_138(c_void_p(buf467.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf474.data_ptr()))
    del arg231_1
    del arg232_1
    buf475 = buf456; del buf456  # reuse
    # Source Nodes: [query_115], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg233_1, (4096, 4096), (1, 4096), 0), out=buf475)
    del arg233_1
    buf476 = buf455; del buf455  # reuse
    # Source Nodes: [key_115], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg234_1, (4096, 4096), (1, 4096), 0), out=buf476)
    del arg234_1
    buf477 = reinterpret_tensor(buf435, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf435  # reuse
    buf478 = reinterpret_tensor(buf434, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf434  # reuse
    cpp_fused_cat_139(c_void_p(buf475.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()))
    del arg354_1
    buf479 = reinterpret_tensor(buf464, (16, 128, 128), (16384, 128, 1), 0); del buf464  # reuse
    # Source Nodes: [attn_weights_161], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf477, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf478, (16, 256, 128), (256, 1, 4096), 0), out=buf479)
    buf480 = buf462; del buf462  # reuse
    buf481 = reinterpret_tensor(buf479, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf479  # reuse
    buf482 = buf460; del buf460  # reuse
    cpp_fused__softmax_div_lift_fresh_where_140(c_void_p(buf481.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(buf482.data_ptr()))
    del arg355_1
    del arg356_1
    buf483 = reinterpret_tensor(buf478, (128, 4096), (4096, 1), 0); del buf478  # reuse
    # Source Nodes: [value_46], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf474, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg235_1, (4096, 4096), (1, 4096), 0), out=buf483)
    del arg235_1
    buf484 = buf481; del buf481  # reuse
    cpp_fused__softmax_141(c_void_p(buf484.data_ptr()), c_void_p(buf482.data_ptr()))
    buf485 = reinterpret_tensor(buf477, (16, 128, 256), (32768, 256, 1), 0); del buf477  # reuse
    # Source Nodes: [attn_output_138], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf484, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf483, (16, 128, 256), (256, 4096, 1), 0), out=buf485)
    buf486 = reinterpret_tensor(buf483, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf483  # reuse
    cpp_fused_clone_142(c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    buf487 = reinterpret_tensor(buf485, (128, 4096), (4096, 1), 0); del buf485  # reuse
    # Source Nodes: [attn_output_140], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf486, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg236_1, (4096, 4096), (1, 4096), 0), out=buf487)
    del arg236_1
    buf488 = reinterpret_tensor(buf469, (128, 16384), (16384, 1), 0); del buf469  # reuse
    # Source Nodes: [hidden_states_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg238_1, reinterpret_tensor(buf474, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg237_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf488)
    del arg237_1
    del arg238_1
    buf489 = reinterpret_tensor(buf488, (1, 128, 16384), (2097152, 16384, 1), 0); del buf488  # reuse
    cpp_fused_add_mul_pow_tanh_143(c_void_p(buf489.data_ptr()))
    buf490 = reinterpret_tensor(buf474, (128, 4096), (4096, 1), 0); del buf474  # reuse
    # Source Nodes: [hidden_states_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg240_1, reinterpret_tensor(buf489, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg239_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf490)
    del arg239_1
    del arg240_1
    buf491 = reinterpret_tensor(buf487, (1, 128, 4096), (524288, 4096, 1), 0); del buf487  # reuse
    buf492 = buf472; del buf472  # reuse
    buf493 = buf471; del buf471  # reuse
    buf495 = reinterpret_tensor(buf486, (1, 128, 4096), (524288, 4096, 1), 0); del buf486  # reuse
    cpp_fused_add_native_layer_norm_144(c_void_p(buf491.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf495.data_ptr()))
    del arg241_1
    del arg242_1
    buf496 = buf490; del buf490  # reuse
    # Source Nodes: [query_120], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg243_1, (4096, 4096), (1, 4096), 0), out=buf496)
    del arg243_1
    buf497 = buf470; del buf470  # reuse
    # Source Nodes: [key_120], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg244_1, (4096, 4096), (1, 4096), 0), out=buf497)
    del arg244_1
    buf498 = reinterpret_tensor(buf467, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf467  # reuse
    buf499 = reinterpret_tensor(buf450, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf450  # reuse
    cpp_fused_cat_145(c_void_p(buf496.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf499.data_ptr()))
    del arg357_1
    buf500 = reinterpret_tensor(buf484, (16, 128, 128), (16384, 128, 1), 0); del buf484  # reuse
    # Source Nodes: [attn_weights_168], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf498, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf499, (16, 256, 128), (256, 1, 4096), 0), out=buf500)
    buf501 = buf482; del buf482  # reuse
    buf502 = reinterpret_tensor(buf500, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf500  # reuse
    buf503 = buf480; del buf480  # reuse
    cpp_fused__softmax_div_lift_fresh_where_146(c_void_p(buf502.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf503.data_ptr()))
    del arg358_1
    del arg359_1
    buf504 = reinterpret_tensor(buf499, (128, 4096), (4096, 1), 0); del buf499  # reuse
    # Source Nodes: [value_48], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf495, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg245_1, (4096, 4096), (1, 4096), 0), out=buf504)
    del arg245_1
    buf505 = buf502; del buf502  # reuse
    cpp_fused__softmax_147(c_void_p(buf505.data_ptr()), c_void_p(buf503.data_ptr()))
    buf506 = reinterpret_tensor(buf498, (16, 128, 256), (32768, 256, 1), 0); del buf498  # reuse
    # Source Nodes: [attn_output_144], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf505, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf504, (16, 128, 256), (256, 4096, 1), 0), out=buf506)
    buf507 = reinterpret_tensor(buf504, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf504  # reuse
    cpp_fused_clone_148(c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()))
    buf508 = reinterpret_tensor(buf506, (128, 4096), (4096, 1), 0); del buf506  # reuse
    # Source Nodes: [attn_output_146], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf507, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg246_1, (4096, 4096), (1, 4096), 0), out=buf508)
    del arg246_1
    buf509 = reinterpret_tensor(buf489, (128, 16384), (16384, 1), 0); del buf489  # reuse
    # Source Nodes: [hidden_states_145], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg248_1, reinterpret_tensor(buf495, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg247_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf509)
    del arg247_1
    del arg248_1
    buf510 = reinterpret_tensor(buf509, (1, 128, 16384), (2097152, 16384, 1), 0); del buf509  # reuse
    cpp_fused_add_mul_pow_tanh_149(c_void_p(buf510.data_ptr()))
    buf511 = reinterpret_tensor(buf495, (128, 4096), (4096, 1), 0); del buf495  # reuse
    # Source Nodes: [hidden_states_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg250_1, reinterpret_tensor(buf510, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg249_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf511)
    del arg249_1
    del arg250_1
    buf512 = buf493; del buf493  # reuse
    buf513 = buf492; del buf492  # reuse
    buf515 = reinterpret_tensor(buf507, (1, 128, 4096), (524288, 4096, 1), 0); del buf507  # reuse
    cpp_fused_add_native_layer_norm_150(c_void_p(buf508.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf515.data_ptr()))
    del arg251_1
    del arg252_1
    buf516 = buf497; del buf497  # reuse
    # Source Nodes: [query_125], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf515, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg253_1, (4096, 4096), (1, 4096), 0), out=buf516)
    del arg253_1
    buf517 = buf496; del buf496  # reuse
    # Source Nodes: [key_125], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf515, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg254_1, (4096, 4096), (1, 4096), 0), out=buf517)
    del arg254_1
    buf518 = reinterpret_tensor(buf476, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf476  # reuse
    buf519 = reinterpret_tensor(buf475, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf475  # reuse
    cpp_fused_cat_151(c_void_p(buf516.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()))
    del arg360_1
    buf520 = reinterpret_tensor(buf505, (16, 128, 128), (16384, 128, 1), 0); del buf505  # reuse
    # Source Nodes: [attn_weights_175], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf518, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf519, (16, 256, 128), (256, 1, 4096), 0), out=buf520)
    buf521 = buf503; del buf503  # reuse
    buf522 = reinterpret_tensor(buf520, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf520  # reuse
    buf523 = buf501; del buf501  # reuse
    cpp_fused__softmax_div_lift_fresh_where_152(c_void_p(buf522.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf523.data_ptr()))
    del arg361_1
    del arg362_1
    buf524 = reinterpret_tensor(buf519, (128, 4096), (4096, 1), 0); del buf519  # reuse
    # Source Nodes: [value_50], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf515, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg255_1, (4096, 4096), (1, 4096), 0), out=buf524)
    del arg255_1
    buf525 = buf522; del buf522  # reuse
    cpp_fused__softmax_153(c_void_p(buf525.data_ptr()), c_void_p(buf523.data_ptr()))
    buf526 = reinterpret_tensor(buf518, (16, 128, 256), (32768, 256, 1), 0); del buf518  # reuse
    # Source Nodes: [attn_output_150], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf525, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf524, (16, 128, 256), (256, 4096, 1), 0), out=buf526)
    buf527 = reinterpret_tensor(buf524, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf524  # reuse
    cpp_fused_clone_154(c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()))
    buf528 = reinterpret_tensor(buf526, (128, 4096), (4096, 1), 0); del buf526  # reuse
    # Source Nodes: [attn_output_152], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf527, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg256_1, (4096, 4096), (1, 4096), 0), out=buf528)
    del arg256_1
    buf529 = reinterpret_tensor(buf510, (128, 16384), (16384, 1), 0); del buf510  # reuse
    # Source Nodes: [hidden_states_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg258_1, reinterpret_tensor(buf515, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg257_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf529)
    del arg257_1
    del arg258_1
    buf530 = reinterpret_tensor(buf529, (1, 128, 16384), (2097152, 16384, 1), 0); del buf529  # reuse
    cpp_fused_add_mul_pow_tanh_155(c_void_p(buf530.data_ptr()))
    buf531 = reinterpret_tensor(buf515, (128, 4096), (4096, 1), 0); del buf515  # reuse
    # Source Nodes: [hidden_states_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg260_1, reinterpret_tensor(buf530, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg259_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf531)
    del arg259_1
    del arg260_1
    buf532 = reinterpret_tensor(buf528, (1, 128, 4096), (524288, 4096, 1), 0); del buf528  # reuse
    buf533 = buf513; del buf513  # reuse
    buf534 = buf512; del buf512  # reuse
    buf536 = reinterpret_tensor(buf527, (1, 128, 4096), (524288, 4096, 1), 0); del buf527  # reuse
    cpp_fused_add_native_layer_norm_156(c_void_p(buf532.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf536.data_ptr()))
    del arg261_1
    del arg262_1
    buf537 = buf531; del buf531  # reuse
    # Source Nodes: [query_130], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf536, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg263_1, (4096, 4096), (1, 4096), 0), out=buf537)
    del arg263_1
    buf538 = buf511; del buf511  # reuse
    # Source Nodes: [key_130], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf536, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg264_1, (4096, 4096), (1, 4096), 0), out=buf538)
    del arg264_1
    buf539 = reinterpret_tensor(buf508, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf508  # reuse
    buf540 = reinterpret_tensor(buf491, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf491  # reuse
    cpp_fused_cat_157(c_void_p(buf537.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf540.data_ptr()))
    del arg363_1
    buf541 = reinterpret_tensor(buf525, (16, 128, 128), (16384, 128, 1), 0); del buf525  # reuse
    # Source Nodes: [attn_weights_182], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf539, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf540, (16, 256, 128), (256, 1, 4096), 0), out=buf541)
    buf542 = buf523; del buf523  # reuse
    buf543 = reinterpret_tensor(buf541, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf541  # reuse
    buf544 = buf521; del buf521  # reuse
    cpp_fused__softmax_div_lift_fresh_where_158(c_void_p(buf543.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf544.data_ptr()))
    del arg364_1
    del arg365_1
    buf545 = reinterpret_tensor(buf540, (128, 4096), (4096, 1), 0); del buf540  # reuse
    # Source Nodes: [value_52], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf536, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg265_1, (4096, 4096), (1, 4096), 0), out=buf545)
    del arg265_1
    buf546 = buf543; del buf543  # reuse
    cpp_fused__softmax_159(c_void_p(buf546.data_ptr()), c_void_p(buf544.data_ptr()))
    buf547 = reinterpret_tensor(buf539, (16, 128, 256), (32768, 256, 1), 0); del buf539  # reuse
    # Source Nodes: [attn_output_156], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf546, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf545, (16, 128, 256), (256, 4096, 1), 0), out=buf547)
    buf548 = reinterpret_tensor(buf545, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf545  # reuse
    cpp_fused_clone_160(c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()))
    buf549 = reinterpret_tensor(buf547, (128, 4096), (4096, 1), 0); del buf547  # reuse
    # Source Nodes: [attn_output_158], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf548, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg266_1, (4096, 4096), (1, 4096), 0), out=buf549)
    del arg266_1
    buf550 = reinterpret_tensor(buf530, (128, 16384), (16384, 1), 0); del buf530  # reuse
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg268_1, reinterpret_tensor(buf536, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg267_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf550)
    del arg267_1
    del arg268_1
    buf551 = reinterpret_tensor(buf550, (1, 128, 16384), (2097152, 16384, 1), 0); del buf550  # reuse
    cpp_fused_add_mul_pow_tanh_161(c_void_p(buf551.data_ptr()))
    buf552 = reinterpret_tensor(buf536, (128, 4096), (4096, 1), 0); del buf536  # reuse
    # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg270_1, reinterpret_tensor(buf551, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg269_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf552)
    del arg269_1
    del arg270_1
    buf553 = buf534; del buf534  # reuse
    buf554 = buf533; del buf533  # reuse
    buf556 = reinterpret_tensor(buf548, (1, 128, 4096), (524288, 4096, 1), 0); del buf548  # reuse
    cpp_fused_add_native_layer_norm_162(c_void_p(buf549.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf556.data_ptr()))
    del arg271_1
    del arg272_1
    buf557 = buf538; del buf538  # reuse
    # Source Nodes: [query_135], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf556, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg273_1, (4096, 4096), (1, 4096), 0), out=buf557)
    del arg273_1
    buf558 = buf537; del buf537  # reuse
    # Source Nodes: [key_135], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf556, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg274_1, (4096, 4096), (1, 4096), 0), out=buf558)
    del arg274_1
    buf559 = reinterpret_tensor(buf517, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf517  # reuse
    buf560 = reinterpret_tensor(buf516, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf516  # reuse
    cpp_fused_cat_163(c_void_p(buf557.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()))
    del arg366_1
    del buf557
    del buf558
    buf561 = reinterpret_tensor(buf546, (16, 128, 128), (16384, 128, 1), 0); del buf546  # reuse
    # Source Nodes: [attn_weights_189], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf559, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf560, (16, 256, 128), (256, 1, 4096), 0), out=buf561)
    buf562 = buf544; del buf544  # reuse
    buf563 = reinterpret_tensor(buf561, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf561  # reuse
    buf564 = buf542; del buf542  # reuse
    cpp_fused__softmax_div_lift_fresh_where_164(c_void_p(buf563.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf564.data_ptr()))
    del arg367_1
    del arg368_1
    del buf562
    buf565 = reinterpret_tensor(buf560, (128, 4096), (4096, 1), 0); del buf560  # reuse
    # Source Nodes: [value_54], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf556, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg275_1, (4096, 4096), (1, 4096), 0), out=buf565)
    del arg275_1
    buf566 = buf563; del buf563  # reuse
    cpp_fused__softmax_165(c_void_p(buf566.data_ptr()), c_void_p(buf564.data_ptr()))
    del buf564
    buf567 = reinterpret_tensor(buf559, (16, 128, 256), (32768, 256, 1), 0); del buf559  # reuse
    # Source Nodes: [attn_output_162], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf566, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf565, (16, 128, 256), (256, 4096, 1), 0), out=buf567)
    del buf566
    buf568 = reinterpret_tensor(buf565, (1, 128, 16, 256), (524288, 4096, 256, 1), 0); del buf565  # reuse
    cpp_fused_clone_166(c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()))
    buf569 = reinterpret_tensor(buf567, (128, 4096), (4096, 1), 0); del buf567  # reuse
    # Source Nodes: [attn_output_164], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf568, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg276_1, (4096, 4096), (1, 4096), 0), out=buf569)
    del arg276_1
    buf570 = reinterpret_tensor(buf551, (128, 16384), (16384, 1), 0); del buf551  # reuse
    # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg278_1, reinterpret_tensor(buf556, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg277_1, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf570)
    del arg277_1
    del arg278_1
    buf571 = reinterpret_tensor(buf570, (1, 128, 16384), (2097152, 16384, 1), 0); del buf570  # reuse
    cpp_fused_add_mul_pow_tanh_167(c_void_p(buf571.data_ptr()))
    buf572 = reinterpret_tensor(buf556, (128, 4096), (4096, 1), 0); del buf556  # reuse
    # Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg280_1, reinterpret_tensor(buf571, (128, 16384), (16384, 1), 0), reinterpret_tensor(arg279_1, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf572)
    del arg279_1
    del arg280_1
    del buf571
    buf573 = reinterpret_tensor(buf569, (1, 128, 4096), (524288, 4096, 1), 0); del buf569  # reuse
    buf574 = buf554; del buf554  # reuse
    buf575 = buf553; del buf553  # reuse
    buf577 = reinterpret_tensor(buf568, (1, 128, 4096), (524288, 4096, 1), 0); del buf568  # reuse
    cpp_fused_add_native_layer_norm_168(c_void_p(buf573.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf577.data_ptr()))
    del arg281_1
    del arg282_1
    del buf532
    del buf549
    del buf552
    del buf572
    del buf573
    buf578 = empty((128, 2), device='cpu', dtype=torch.float32)
    # Source Nodes: [logits], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg284_1, reinterpret_tensor(buf577, (128, 4096), (4096, 1), 0), reinterpret_tensor(arg283_1, (4096, 2), (1, 4096), 0), alpha=1, beta=1, out=buf578)
    del arg283_1
    del arg284_1
    del buf577
    buf579 = reinterpret_tensor(buf575, (1, 128), (128, 1), 0); del buf575  # reuse
    buf580 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf581 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf582 = reinterpret_tensor(buf574, (1, 128), (128, 1), 0); del buf574  # reuse
    buf583 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf584 = empty((1, 1), device='cpu', dtype=torch.float32)
    buf585 = reinterpret_tensor(buf580, (), (), 0); del buf580  # reuse
    buf586 = buf585; del buf585  # reuse
    cpp_fused__log_softmax_add_clamp_clone_div_nll_loss_forward_169(c_void_p(buf586.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()))
    del arg370_1
    del arg371_1
    return (buf586, buf579, buf582, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((50400, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((2, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg287_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg290_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg293_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg296_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg299_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg302_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg305_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg308_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg311_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg314_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg317_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg320_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg323_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg326_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg329_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg332_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg335_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg338_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg341_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg344_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg347_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg350_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg353_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg356_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg359_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg362_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg365_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    arg368_1 = rand_strided((), (), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg370_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    arg371_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPTJForQuestionAnswering', benchmark_compiled_module)
