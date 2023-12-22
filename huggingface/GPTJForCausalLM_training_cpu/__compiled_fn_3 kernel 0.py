
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


cpp_fused_embedding_native_layer_norm_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
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
                        tmp4.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_1 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_2 = async_compile.cpp('''
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


cpp_fused_view_3 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_5 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_6 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_7 = async_compile.cpp('''
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


cpp_fused_view_8 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_11 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_12 = async_compile.cpp('''
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


cpp_fused_view_13 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_16 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_17 = async_compile.cpp('''
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


cpp_fused_view_18 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_21 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_22 = async_compile.cpp('''
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


cpp_fused_view_23 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_26 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_27 = async_compile.cpp('''
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


cpp_fused_view_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_31 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_32 = async_compile.cpp('''
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


cpp_fused_view_33 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_36 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_37 = async_compile.cpp('''
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


cpp_fused_view_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_41 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_42 = async_compile.cpp('''
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


cpp_fused_view_43 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_46 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_47 = async_compile.cpp('''
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


cpp_fused_view_48 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_51 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_52 = async_compile.cpp('''
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


cpp_fused_view_53 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_56 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_57 = async_compile.cpp('''
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


cpp_fused_view_58 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_61 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_62 = async_compile.cpp('''
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


cpp_fused_view_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_66 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_67 = async_compile.cpp('''
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


cpp_fused_view_68 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_71 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_72 = async_compile.cpp('''
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


cpp_fused_view_73 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_76 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_77 = async_compile.cpp('''
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


cpp_fused_view_78 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_81 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_82 = async_compile.cpp('''
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


cpp_fused_view_83 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_86 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_87 = async_compile.cpp('''
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


cpp_fused_view_88 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_91 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_92 = async_compile.cpp('''
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


cpp_fused_view_93 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_96 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_97 = async_compile.cpp('''
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


cpp_fused_view_98 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_101 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_102 = async_compile.cpp('''
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


cpp_fused_view_103 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_106 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_107 = async_compile.cpp('''
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


cpp_fused_view_108 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_111 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_112 = async_compile.cpp('''
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


cpp_fused_view_113 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_116 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_117 = async_compile.cpp('''
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


cpp_fused_view_118 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_121 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_122 = async_compile.cpp('''
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


cpp_fused_view_123 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_125 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_126 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_lift_fresh_where_127 = async_compile.cpp('''
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


cpp_fused_view_128 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_130 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_131 = async_compile.cpp('''
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


cpp_fused__softmax_clone_detach_div_lift_fresh_where_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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


cpp_fused_view_133 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_134 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_135 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_cat_permute_136 = async_compile.cpp('''
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


cpp_fused__softmax_clone_detach_div_lift_fresh_where_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
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


cpp_fused_view_138 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4096L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((256L*x0) + (32768L*(c10::div_floor_integer((x1 + x1_inner), 256L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(256L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_pow_tanh_view_139 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2097152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_140 = async_compile.cpp('''
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (4096L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (4096L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__log_softmax__softmax_add_detach_gather_native_layer_norm_native_layer_norm_backward_nll_loss_forward_141 = async_compile.cpp('''
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
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr3,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17,
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25,
                       float* out_ptr26,
                       float* out_ptr27,
                       float* out_ptr28,
                       float* out_ptr29,
                       float* out_ptr30,
                       float* out_ptr31,
                       float* out_ptr32,
                       float* out_ptr33)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50400L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50400L*x0)));
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50400L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50400L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(x0)];
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp4 = tmp3.exp();
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50400L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (50400L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = std::log(tmp4);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp3 - tmp6;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (50400L*x0)));
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
                        auto tmp6 = decltype(tmp5)(tmp5 + 50400);
                        auto tmp7 = tmp5 < 0;
                        auto tmp8 = tmp7 ? tmp6 : tmp5;
                        TORCH_CHECK((0 <= tmp8) & (tmp8 < 50400L), "index out of bounds: 0 <= tmp8 < 50400L")
                        auto tmp9 = out_ptr2[static_cast<long>(tmp8 + (50400L*x0))];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr3[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr6 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr8 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr12 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr14 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr18 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr20 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr11[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr24 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr13[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr26 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr14[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr15[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr30 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr16[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr32 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr17[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr18[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr36 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr37 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr19[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr38 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr20[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr21[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr42 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr43 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr22[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr44 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr23[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr24[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr48 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr49 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr25[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr50 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(4096.0);
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
                    auto tmp1 = in_ptr26[static_cast<long>(x0)];
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
                    auto tmp1 = static_cast<float>(4096.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr27[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr54 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr28 + static_cast<long>(x0));
                    tmp0.store(out_ptr6 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x0));
                    tmp0.store(out_ptr7 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x0));
                    tmp0.store(out_ptr8 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x0));
                    tmp0.store(out_ptr9 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr32 + static_cast<long>(x0));
                    tmp0.store(out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x0));
                    tmp0.store(out_ptr11 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr34 + static_cast<long>(x0));
                    tmp0.store(out_ptr12 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x0));
                    tmp0.store(out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr36 + static_cast<long>(x0));
                    tmp0.store(out_ptr14 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x0));
                    tmp0.store(out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr38 + static_cast<long>(x0));
                    tmp0.store(out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x0));
                    tmp0.store(out_ptr17 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr40 + static_cast<long>(x0));
                    tmp0.store(out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x0));
                    tmp0.store(out_ptr19 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr42 + static_cast<long>(x0));
                    tmp0.store(out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x0));
                    tmp0.store(out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr44 + static_cast<long>(x0));
                    tmp0.store(out_ptr22 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x0));
                    tmp0.store(out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x0));
                    tmp0.store(out_ptr24 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x0));
                    tmp0.store(out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr48 + static_cast<long>(x0));
                    tmp0.store(out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr49 + static_cast<long>(x0));
                    tmp0.store(out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr50 + static_cast<long>(x0));
                    tmp0.store(out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr51 + static_cast<long>(x0));
                    tmp0.store(out_ptr29 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr52 + static_cast<long>(x0));
                    tmp0.store(out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr53 + static_cast<long>(x0));
                    tmp0.store(out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr54 + static_cast<long>(x0));
                    tmp0.store(out_ptr32 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr55 + static_cast<long>(x0));
                    tmp0.store(out_ptr33 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371 = args
    args.clear()
    assert_size_stride(primals_1, (50400, 4096), (4096, 1))
    assert_size_stride(primals_2, (4096, ), (1, ))
    assert_size_stride(primals_3, (4096, ), (1, ))
    assert_size_stride(primals_4, (4096, 4096), (4096, 1))
    assert_size_stride(primals_5, (4096, 4096), (4096, 1))
    assert_size_stride(primals_6, (4096, 4096), (4096, 1))
    assert_size_stride(primals_7, (4096, 4096), (4096, 1))
    assert_size_stride(primals_8, (16384, 4096), (4096, 1))
    assert_size_stride(primals_9, (16384, ), (1, ))
    assert_size_stride(primals_10, (4096, 16384), (16384, 1))
    assert_size_stride(primals_11, (4096, ), (1, ))
    assert_size_stride(primals_12, (4096, ), (1, ))
    assert_size_stride(primals_13, (4096, ), (1, ))
    assert_size_stride(primals_14, (4096, 4096), (4096, 1))
    assert_size_stride(primals_15, (4096, 4096), (4096, 1))
    assert_size_stride(primals_16, (4096, 4096), (4096, 1))
    assert_size_stride(primals_17, (4096, 4096), (4096, 1))
    assert_size_stride(primals_18, (16384, 4096), (4096, 1))
    assert_size_stride(primals_19, (16384, ), (1, ))
    assert_size_stride(primals_20, (4096, 16384), (16384, 1))
    assert_size_stride(primals_21, (4096, ), (1, ))
    assert_size_stride(primals_22, (4096, ), (1, ))
    assert_size_stride(primals_23, (4096, ), (1, ))
    assert_size_stride(primals_24, (4096, 4096), (4096, 1))
    assert_size_stride(primals_25, (4096, 4096), (4096, 1))
    assert_size_stride(primals_26, (4096, 4096), (4096, 1))
    assert_size_stride(primals_27, (4096, 4096), (4096, 1))
    assert_size_stride(primals_28, (16384, 4096), (4096, 1))
    assert_size_stride(primals_29, (16384, ), (1, ))
    assert_size_stride(primals_30, (4096, 16384), (16384, 1))
    assert_size_stride(primals_31, (4096, ), (1, ))
    assert_size_stride(primals_32, (4096, ), (1, ))
    assert_size_stride(primals_33, (4096, ), (1, ))
    assert_size_stride(primals_34, (4096, 4096), (4096, 1))
    assert_size_stride(primals_35, (4096, 4096), (4096, 1))
    assert_size_stride(primals_36, (4096, 4096), (4096, 1))
    assert_size_stride(primals_37, (4096, 4096), (4096, 1))
    assert_size_stride(primals_38, (16384, 4096), (4096, 1))
    assert_size_stride(primals_39, (16384, ), (1, ))
    assert_size_stride(primals_40, (4096, 16384), (16384, 1))
    assert_size_stride(primals_41, (4096, ), (1, ))
    assert_size_stride(primals_42, (4096, ), (1, ))
    assert_size_stride(primals_43, (4096, ), (1, ))
    assert_size_stride(primals_44, (4096, 4096), (4096, 1))
    assert_size_stride(primals_45, (4096, 4096), (4096, 1))
    assert_size_stride(primals_46, (4096, 4096), (4096, 1))
    assert_size_stride(primals_47, (4096, 4096), (4096, 1))
    assert_size_stride(primals_48, (16384, 4096), (4096, 1))
    assert_size_stride(primals_49, (16384, ), (1, ))
    assert_size_stride(primals_50, (4096, 16384), (16384, 1))
    assert_size_stride(primals_51, (4096, ), (1, ))
    assert_size_stride(primals_52, (4096, ), (1, ))
    assert_size_stride(primals_53, (4096, ), (1, ))
    assert_size_stride(primals_54, (4096, 4096), (4096, 1))
    assert_size_stride(primals_55, (4096, 4096), (4096, 1))
    assert_size_stride(primals_56, (4096, 4096), (4096, 1))
    assert_size_stride(primals_57, (4096, 4096), (4096, 1))
    assert_size_stride(primals_58, (16384, 4096), (4096, 1))
    assert_size_stride(primals_59, (16384, ), (1, ))
    assert_size_stride(primals_60, (4096, 16384), (16384, 1))
    assert_size_stride(primals_61, (4096, ), (1, ))
    assert_size_stride(primals_62, (4096, ), (1, ))
    assert_size_stride(primals_63, (4096, ), (1, ))
    assert_size_stride(primals_64, (4096, 4096), (4096, 1))
    assert_size_stride(primals_65, (4096, 4096), (4096, 1))
    assert_size_stride(primals_66, (4096, 4096), (4096, 1))
    assert_size_stride(primals_67, (4096, 4096), (4096, 1))
    assert_size_stride(primals_68, (16384, 4096), (4096, 1))
    assert_size_stride(primals_69, (16384, ), (1, ))
    assert_size_stride(primals_70, (4096, 16384), (16384, 1))
    assert_size_stride(primals_71, (4096, ), (1, ))
    assert_size_stride(primals_72, (4096, ), (1, ))
    assert_size_stride(primals_73, (4096, ), (1, ))
    assert_size_stride(primals_74, (4096, 4096), (4096, 1))
    assert_size_stride(primals_75, (4096, 4096), (4096, 1))
    assert_size_stride(primals_76, (4096, 4096), (4096, 1))
    assert_size_stride(primals_77, (4096, 4096), (4096, 1))
    assert_size_stride(primals_78, (16384, 4096), (4096, 1))
    assert_size_stride(primals_79, (16384, ), (1, ))
    assert_size_stride(primals_80, (4096, 16384), (16384, 1))
    assert_size_stride(primals_81, (4096, ), (1, ))
    assert_size_stride(primals_82, (4096, ), (1, ))
    assert_size_stride(primals_83, (4096, ), (1, ))
    assert_size_stride(primals_84, (4096, 4096), (4096, 1))
    assert_size_stride(primals_85, (4096, 4096), (4096, 1))
    assert_size_stride(primals_86, (4096, 4096), (4096, 1))
    assert_size_stride(primals_87, (4096, 4096), (4096, 1))
    assert_size_stride(primals_88, (16384, 4096), (4096, 1))
    assert_size_stride(primals_89, (16384, ), (1, ))
    assert_size_stride(primals_90, (4096, 16384), (16384, 1))
    assert_size_stride(primals_91, (4096, ), (1, ))
    assert_size_stride(primals_92, (4096, ), (1, ))
    assert_size_stride(primals_93, (4096, ), (1, ))
    assert_size_stride(primals_94, (4096, 4096), (4096, 1))
    assert_size_stride(primals_95, (4096, 4096), (4096, 1))
    assert_size_stride(primals_96, (4096, 4096), (4096, 1))
    assert_size_stride(primals_97, (4096, 4096), (4096, 1))
    assert_size_stride(primals_98, (16384, 4096), (4096, 1))
    assert_size_stride(primals_99, (16384, ), (1, ))
    assert_size_stride(primals_100, (4096, 16384), (16384, 1))
    assert_size_stride(primals_101, (4096, ), (1, ))
    assert_size_stride(primals_102, (4096, ), (1, ))
    assert_size_stride(primals_103, (4096, ), (1, ))
    assert_size_stride(primals_104, (4096, 4096), (4096, 1))
    assert_size_stride(primals_105, (4096, 4096), (4096, 1))
    assert_size_stride(primals_106, (4096, 4096), (4096, 1))
    assert_size_stride(primals_107, (4096, 4096), (4096, 1))
    assert_size_stride(primals_108, (16384, 4096), (4096, 1))
    assert_size_stride(primals_109, (16384, ), (1, ))
    assert_size_stride(primals_110, (4096, 16384), (16384, 1))
    assert_size_stride(primals_111, (4096, ), (1, ))
    assert_size_stride(primals_112, (4096, ), (1, ))
    assert_size_stride(primals_113, (4096, ), (1, ))
    assert_size_stride(primals_114, (4096, 4096), (4096, 1))
    assert_size_stride(primals_115, (4096, 4096), (4096, 1))
    assert_size_stride(primals_116, (4096, 4096), (4096, 1))
    assert_size_stride(primals_117, (4096, 4096), (4096, 1))
    assert_size_stride(primals_118, (16384, 4096), (4096, 1))
    assert_size_stride(primals_119, (16384, ), (1, ))
    assert_size_stride(primals_120, (4096, 16384), (16384, 1))
    assert_size_stride(primals_121, (4096, ), (1, ))
    assert_size_stride(primals_122, (4096, ), (1, ))
    assert_size_stride(primals_123, (4096, ), (1, ))
    assert_size_stride(primals_124, (4096, 4096), (4096, 1))
    assert_size_stride(primals_125, (4096, 4096), (4096, 1))
    assert_size_stride(primals_126, (4096, 4096), (4096, 1))
    assert_size_stride(primals_127, (4096, 4096), (4096, 1))
    assert_size_stride(primals_128, (16384, 4096), (4096, 1))
    assert_size_stride(primals_129, (16384, ), (1, ))
    assert_size_stride(primals_130, (4096, 16384), (16384, 1))
    assert_size_stride(primals_131, (4096, ), (1, ))
    assert_size_stride(primals_132, (4096, ), (1, ))
    assert_size_stride(primals_133, (4096, ), (1, ))
    assert_size_stride(primals_134, (4096, 4096), (4096, 1))
    assert_size_stride(primals_135, (4096, 4096), (4096, 1))
    assert_size_stride(primals_136, (4096, 4096), (4096, 1))
    assert_size_stride(primals_137, (4096, 4096), (4096, 1))
    assert_size_stride(primals_138, (16384, 4096), (4096, 1))
    assert_size_stride(primals_139, (16384, ), (1, ))
    assert_size_stride(primals_140, (4096, 16384), (16384, 1))
    assert_size_stride(primals_141, (4096, ), (1, ))
    assert_size_stride(primals_142, (4096, ), (1, ))
    assert_size_stride(primals_143, (4096, ), (1, ))
    assert_size_stride(primals_144, (4096, 4096), (4096, 1))
    assert_size_stride(primals_145, (4096, 4096), (4096, 1))
    assert_size_stride(primals_146, (4096, 4096), (4096, 1))
    assert_size_stride(primals_147, (4096, 4096), (4096, 1))
    assert_size_stride(primals_148, (16384, 4096), (4096, 1))
    assert_size_stride(primals_149, (16384, ), (1, ))
    assert_size_stride(primals_150, (4096, 16384), (16384, 1))
    assert_size_stride(primals_151, (4096, ), (1, ))
    assert_size_stride(primals_152, (4096, ), (1, ))
    assert_size_stride(primals_153, (4096, ), (1, ))
    assert_size_stride(primals_154, (4096, 4096), (4096, 1))
    assert_size_stride(primals_155, (4096, 4096), (4096, 1))
    assert_size_stride(primals_156, (4096, 4096), (4096, 1))
    assert_size_stride(primals_157, (4096, 4096), (4096, 1))
    assert_size_stride(primals_158, (16384, 4096), (4096, 1))
    assert_size_stride(primals_159, (16384, ), (1, ))
    assert_size_stride(primals_160, (4096, 16384), (16384, 1))
    assert_size_stride(primals_161, (4096, ), (1, ))
    assert_size_stride(primals_162, (4096, ), (1, ))
    assert_size_stride(primals_163, (4096, ), (1, ))
    assert_size_stride(primals_164, (4096, 4096), (4096, 1))
    assert_size_stride(primals_165, (4096, 4096), (4096, 1))
    assert_size_stride(primals_166, (4096, 4096), (4096, 1))
    assert_size_stride(primals_167, (4096, 4096), (4096, 1))
    assert_size_stride(primals_168, (16384, 4096), (4096, 1))
    assert_size_stride(primals_169, (16384, ), (1, ))
    assert_size_stride(primals_170, (4096, 16384), (16384, 1))
    assert_size_stride(primals_171, (4096, ), (1, ))
    assert_size_stride(primals_172, (4096, ), (1, ))
    assert_size_stride(primals_173, (4096, ), (1, ))
    assert_size_stride(primals_174, (4096, 4096), (4096, 1))
    assert_size_stride(primals_175, (4096, 4096), (4096, 1))
    assert_size_stride(primals_176, (4096, 4096), (4096, 1))
    assert_size_stride(primals_177, (4096, 4096), (4096, 1))
    assert_size_stride(primals_178, (16384, 4096), (4096, 1))
    assert_size_stride(primals_179, (16384, ), (1, ))
    assert_size_stride(primals_180, (4096, 16384), (16384, 1))
    assert_size_stride(primals_181, (4096, ), (1, ))
    assert_size_stride(primals_182, (4096, ), (1, ))
    assert_size_stride(primals_183, (4096, ), (1, ))
    assert_size_stride(primals_184, (4096, 4096), (4096, 1))
    assert_size_stride(primals_185, (4096, 4096), (4096, 1))
    assert_size_stride(primals_186, (4096, 4096), (4096, 1))
    assert_size_stride(primals_187, (4096, 4096), (4096, 1))
    assert_size_stride(primals_188, (16384, 4096), (4096, 1))
    assert_size_stride(primals_189, (16384, ), (1, ))
    assert_size_stride(primals_190, (4096, 16384), (16384, 1))
    assert_size_stride(primals_191, (4096, ), (1, ))
    assert_size_stride(primals_192, (4096, ), (1, ))
    assert_size_stride(primals_193, (4096, ), (1, ))
    assert_size_stride(primals_194, (4096, 4096), (4096, 1))
    assert_size_stride(primals_195, (4096, 4096), (4096, 1))
    assert_size_stride(primals_196, (4096, 4096), (4096, 1))
    assert_size_stride(primals_197, (4096, 4096), (4096, 1))
    assert_size_stride(primals_198, (16384, 4096), (4096, 1))
    assert_size_stride(primals_199, (16384, ), (1, ))
    assert_size_stride(primals_200, (4096, 16384), (16384, 1))
    assert_size_stride(primals_201, (4096, ), (1, ))
    assert_size_stride(primals_202, (4096, ), (1, ))
    assert_size_stride(primals_203, (4096, ), (1, ))
    assert_size_stride(primals_204, (4096, 4096), (4096, 1))
    assert_size_stride(primals_205, (4096, 4096), (4096, 1))
    assert_size_stride(primals_206, (4096, 4096), (4096, 1))
    assert_size_stride(primals_207, (4096, 4096), (4096, 1))
    assert_size_stride(primals_208, (16384, 4096), (4096, 1))
    assert_size_stride(primals_209, (16384, ), (1, ))
    assert_size_stride(primals_210, (4096, 16384), (16384, 1))
    assert_size_stride(primals_211, (4096, ), (1, ))
    assert_size_stride(primals_212, (4096, ), (1, ))
    assert_size_stride(primals_213, (4096, ), (1, ))
    assert_size_stride(primals_214, (4096, 4096), (4096, 1))
    assert_size_stride(primals_215, (4096, 4096), (4096, 1))
    assert_size_stride(primals_216, (4096, 4096), (4096, 1))
    assert_size_stride(primals_217, (4096, 4096), (4096, 1))
    assert_size_stride(primals_218, (16384, 4096), (4096, 1))
    assert_size_stride(primals_219, (16384, ), (1, ))
    assert_size_stride(primals_220, (4096, 16384), (16384, 1))
    assert_size_stride(primals_221, (4096, ), (1, ))
    assert_size_stride(primals_222, (4096, ), (1, ))
    assert_size_stride(primals_223, (4096, ), (1, ))
    assert_size_stride(primals_224, (4096, 4096), (4096, 1))
    assert_size_stride(primals_225, (4096, 4096), (4096, 1))
    assert_size_stride(primals_226, (4096, 4096), (4096, 1))
    assert_size_stride(primals_227, (4096, 4096), (4096, 1))
    assert_size_stride(primals_228, (16384, 4096), (4096, 1))
    assert_size_stride(primals_229, (16384, ), (1, ))
    assert_size_stride(primals_230, (4096, 16384), (16384, 1))
    assert_size_stride(primals_231, (4096, ), (1, ))
    assert_size_stride(primals_232, (4096, ), (1, ))
    assert_size_stride(primals_233, (4096, ), (1, ))
    assert_size_stride(primals_234, (4096, 4096), (4096, 1))
    assert_size_stride(primals_235, (4096, 4096), (4096, 1))
    assert_size_stride(primals_236, (4096, 4096), (4096, 1))
    assert_size_stride(primals_237, (4096, 4096), (4096, 1))
    assert_size_stride(primals_238, (16384, 4096), (4096, 1))
    assert_size_stride(primals_239, (16384, ), (1, ))
    assert_size_stride(primals_240, (4096, 16384), (16384, 1))
    assert_size_stride(primals_241, (4096, ), (1, ))
    assert_size_stride(primals_242, (4096, ), (1, ))
    assert_size_stride(primals_243, (4096, ), (1, ))
    assert_size_stride(primals_244, (4096, 4096), (4096, 1))
    assert_size_stride(primals_245, (4096, 4096), (4096, 1))
    assert_size_stride(primals_246, (4096, 4096), (4096, 1))
    assert_size_stride(primals_247, (4096, 4096), (4096, 1))
    assert_size_stride(primals_248, (16384, 4096), (4096, 1))
    assert_size_stride(primals_249, (16384, ), (1, ))
    assert_size_stride(primals_250, (4096, 16384), (16384, 1))
    assert_size_stride(primals_251, (4096, ), (1, ))
    assert_size_stride(primals_252, (4096, ), (1, ))
    assert_size_stride(primals_253, (4096, ), (1, ))
    assert_size_stride(primals_254, (4096, 4096), (4096, 1))
    assert_size_stride(primals_255, (4096, 4096), (4096, 1))
    assert_size_stride(primals_256, (4096, 4096), (4096, 1))
    assert_size_stride(primals_257, (4096, 4096), (4096, 1))
    assert_size_stride(primals_258, (16384, 4096), (4096, 1))
    assert_size_stride(primals_259, (16384, ), (1, ))
    assert_size_stride(primals_260, (4096, 16384), (16384, 1))
    assert_size_stride(primals_261, (4096, ), (1, ))
    assert_size_stride(primals_262, (4096, ), (1, ))
    assert_size_stride(primals_263, (4096, ), (1, ))
    assert_size_stride(primals_264, (4096, 4096), (4096, 1))
    assert_size_stride(primals_265, (4096, 4096), (4096, 1))
    assert_size_stride(primals_266, (4096, 4096), (4096, 1))
    assert_size_stride(primals_267, (4096, 4096), (4096, 1))
    assert_size_stride(primals_268, (16384, 4096), (4096, 1))
    assert_size_stride(primals_269, (16384, ), (1, ))
    assert_size_stride(primals_270, (4096, 16384), (16384, 1))
    assert_size_stride(primals_271, (4096, ), (1, ))
    assert_size_stride(primals_272, (4096, ), (1, ))
    assert_size_stride(primals_273, (4096, ), (1, ))
    assert_size_stride(primals_274, (4096, 4096), (4096, 1))
    assert_size_stride(primals_275, (4096, 4096), (4096, 1))
    assert_size_stride(primals_276, (4096, 4096), (4096, 1))
    assert_size_stride(primals_277, (4096, 4096), (4096, 1))
    assert_size_stride(primals_278, (16384, 4096), (4096, 1))
    assert_size_stride(primals_279, (16384, ), (1, ))
    assert_size_stride(primals_280, (4096, 16384), (16384, 1))
    assert_size_stride(primals_281, (4096, ), (1, ))
    assert_size_stride(primals_282, (4096, ), (1, ))
    assert_size_stride(primals_283, (4096, ), (1, ))
    assert_size_stride(primals_284, (50400, 4096), (4096, 1))
    assert_size_stride(primals_285, (50400, ), (1, ))
    assert_size_stride(primals_286, (2048, 64), (64, 1))
    assert_size_stride(primals_287, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (2048, 64), (64, 1))
    assert_size_stride(primals_290, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (2048, 64), (64, 1))
    assert_size_stride(primals_293, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (2048, 64), (64, 1))
    assert_size_stride(primals_296, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (2048, 64), (64, 1))
    assert_size_stride(primals_299, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (2048, 64), (64, 1))
    assert_size_stride(primals_302, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (2048, 64), (64, 1))
    assert_size_stride(primals_305, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (2048, 64), (64, 1))
    assert_size_stride(primals_308, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (2048, 64), (64, 1))
    assert_size_stride(primals_311, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (2048, 64), (64, 1))
    assert_size_stride(primals_314, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (2048, 64), (64, 1))
    assert_size_stride(primals_317, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (2048, 64), (64, 1))
    assert_size_stride(primals_320, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (2048, 64), (64, 1))
    assert_size_stride(primals_323, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (2048, 64), (64, 1))
    assert_size_stride(primals_326, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (2048, 64), (64, 1))
    assert_size_stride(primals_329, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (2048, 64), (64, 1))
    assert_size_stride(primals_332, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (2048, 64), (64, 1))
    assert_size_stride(primals_335, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (2048, 64), (64, 1))
    assert_size_stride(primals_338, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (2048, 64), (64, 1))
    assert_size_stride(primals_341, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (2048, 64), (64, 1))
    assert_size_stride(primals_344, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (2048, 64), (64, 1))
    assert_size_stride(primals_347, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (2048, 64), (64, 1))
    assert_size_stride(primals_350, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (2048, 64), (64, 1))
    assert_size_stride(primals_353, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (2048, 64), (64, 1))
    assert_size_stride(primals_356, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (2048, 64), (64, 1))
    assert_size_stride(primals_359, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (2048, 64), (64, 1))
    assert_size_stride(primals_362, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (2048, 64), (64, 1))
    assert_size_stride(primals_365, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (2048, 64), (64, 1))
    assert_size_stride(primals_368, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (1, 128), (128, 1))
    assert_size_stride(primals_371, (1, 128), (128, 1))
    buf0 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf1 = empty((1, 128, 1), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf4 = reinterpret_tensor(buf2, (1, 128, 1), (128, 1, 1), 0); del buf2  # reuse
    buf5 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_embedding_native_layer_norm_view_0(c_void_p(buf4.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_1
    del primals_3
    buf6 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, reinterpret_tensor(primals_4, (4096, 4096), (1, 4096), 0), out=buf6)
    buf7 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, reinterpret_tensor(primals_5, (4096, 4096), (1, 4096), 0), out=buf7)
    buf8 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value], Original ATen: [aten.mm]
    extern_kernels.mm(buf5, reinterpret_tensor(primals_6, (4096, 4096), (1, 4096), 0), out=buf8)
    buf9 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf10 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_1(c_void_p(buf7.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    buf11 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf10, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf9, (16, 256, 128), (256, 1, 4096), 0), out=buf11)
    buf12 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf11, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf11  # reuse
    buf14 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf15 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_2(c_void_p(buf13.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = reinterpret_tensor(buf7, (16, 128, 256), (32768, 256, 1), 0); del buf7  # reuse
    # Source Nodes: [attn_output], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf15, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf8, (16, 128, 256), (256, 4096, 1), 0), out=buf16)
    buf17 = buf6; del buf6  # reuse
    cpp_fused_view_3(c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = reinterpret_tensor(buf16, (128, 4096), (4096, 1), 0); del buf16  # reuse
    # Source Nodes: [attn_output_2], Original ATen: [aten.mm]
    extern_kernels.mm(buf17, reinterpret_tensor(primals_7, (4096, 4096), (1, 4096), 0), out=buf18)
    buf19 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_9, buf5, reinterpret_tensor(primals_8, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf19)
    del primals_9
    buf20 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf21 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_4(c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_3], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_11, buf21, reinterpret_tensor(primals_10, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf22)
    del primals_11
    buf23 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf26 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf27 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_5(c_void_p(buf18.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del primals_13
    buf28 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf27, reinterpret_tensor(primals_14, (4096, 4096), (1, 4096), 0), out=buf28)
    buf29 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf27, reinterpret_tensor(primals_15, (4096, 4096), (1, 4096), 0), out=buf29)
    buf30 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_2], Original ATen: [aten.mm]
    extern_kernels.mm(buf27, reinterpret_tensor(primals_16, (4096, 4096), (1, 4096), 0), out=buf30)
    buf31 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf32 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_6(c_void_p(buf29.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    buf33 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf32, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf31, (16, 256, 128), (256, 1, 4096), 0), out=buf33)
    buf34 = buf12; del buf12  # reuse
    buf35 = reinterpret_tensor(buf33, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf33  # reuse
    buf36 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf37 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_7(c_void_p(buf35.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = reinterpret_tensor(buf29, (16, 128, 256), (32768, 256, 1), 0); del buf29  # reuse
    # Source Nodes: [attn_output_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf37, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf30, (16, 128, 256), (256, 4096, 1), 0), out=buf38)
    buf39 = buf28; del buf28  # reuse
    cpp_fused_view_8(c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = reinterpret_tensor(buf38, (128, 4096), (4096, 1), 0); del buf38  # reuse
    # Source Nodes: [attn_output_8], Original ATen: [aten.mm]
    extern_kernels.mm(buf39, reinterpret_tensor(primals_17, (4096, 4096), (1, 4096), 0), out=buf40)
    buf41 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_7], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_19, buf27, reinterpret_tensor(primals_18, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf41)
    del primals_19
    buf42 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf43 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_9(c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    buf44 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_9], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_21, buf43, reinterpret_tensor(primals_20, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf44)
    del primals_21
    buf45 = reinterpret_tensor(buf40, (1, 128, 4096), (524288, 4096, 1), 0); del buf40  # reuse
    buf46 = buf23; del buf23  # reuse
    buf47 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf49 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf50 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_10(c_void_p(buf45.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    del primals_23
    buf51 = buf44; del buf44  # reuse
    # Source Nodes: [query_10], Original ATen: [aten.mm]
    extern_kernels.mm(buf50, reinterpret_tensor(primals_24, (4096, 4096), (1, 4096), 0), out=buf51)
    buf52 = buf22; del buf22  # reuse
    # Source Nodes: [key_10], Original ATen: [aten.mm]
    extern_kernels.mm(buf50, reinterpret_tensor(primals_25, (4096, 4096), (1, 4096), 0), out=buf52)
    buf53 = buf18; del buf18  # reuse
    # Source Nodes: [value_4], Original ATen: [aten.mm]
    extern_kernels.mm(buf50, reinterpret_tensor(primals_26, (4096, 4096), (1, 4096), 0), out=buf53)
    buf54 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf55 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_11(c_void_p(buf52.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    buf56 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf55, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf54, (16, 256, 128), (256, 1, 4096), 0), out=buf56)
    buf57 = buf34; del buf34  # reuse
    buf58 = reinterpret_tensor(buf56, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf56  # reuse
    buf59 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf60 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_12(c_void_p(buf58.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    buf61 = reinterpret_tensor(buf52, (16, 128, 256), (32768, 256, 1), 0); del buf52  # reuse
    # Source Nodes: [attn_output_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf60, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf53, (16, 128, 256), (256, 4096, 1), 0), out=buf61)
    buf62 = buf51; del buf51  # reuse
    cpp_fused_view_13(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    buf63 = reinterpret_tensor(buf61, (128, 4096), (4096, 1), 0); del buf61  # reuse
    # Source Nodes: [attn_output_14], Original ATen: [aten.mm]
    extern_kernels.mm(buf62, reinterpret_tensor(primals_27, (4096, 4096), (1, 4096), 0), out=buf63)
    buf64 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_13], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_29, buf50, reinterpret_tensor(primals_28, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf64)
    del primals_29
    buf65 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf66 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_14(c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()))
    buf67 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_31, buf66, reinterpret_tensor(primals_30, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf67)
    del primals_31
    buf68 = buf46; del buf46  # reuse
    buf69 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf71 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf72 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_15(c_void_p(buf63.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    del primals_33
    buf73 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf72, reinterpret_tensor(primals_34, (4096, 4096), (1, 4096), 0), out=buf73)
    buf74 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf72, reinterpret_tensor(primals_35, (4096, 4096), (1, 4096), 0), out=buf74)
    buf75 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_6], Original ATen: [aten.mm]
    extern_kernels.mm(buf72, reinterpret_tensor(primals_36, (4096, 4096), (1, 4096), 0), out=buf75)
    buf76 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf77 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_16(c_void_p(buf74.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()))
    buf78 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf77, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf76, (16, 256, 128), (256, 1, 4096), 0), out=buf78)
    buf79 = buf57; del buf57  # reuse
    buf80 = reinterpret_tensor(buf78, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf78  # reuse
    buf81 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf82 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_17(c_void_p(buf80.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    buf83 = reinterpret_tensor(buf74, (16, 128, 256), (32768, 256, 1), 0); del buf74  # reuse
    # Source Nodes: [attn_output_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf82, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf75, (16, 128, 256), (256, 4096, 1), 0), out=buf83)
    buf84 = buf73; del buf73  # reuse
    cpp_fused_view_18(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf83, (128, 4096), (4096, 1), 0); del buf83  # reuse
    # Source Nodes: [attn_output_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf84, reinterpret_tensor(primals_37, (4096, 4096), (1, 4096), 0), out=buf85)
    buf86 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_39, buf72, reinterpret_tensor(primals_38, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf86)
    del primals_39
    buf87 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf88 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_19(c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()))
    buf89 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_21], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_41, buf88, reinterpret_tensor(primals_40, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf89)
    del primals_41
    buf90 = reinterpret_tensor(buf85, (1, 128, 4096), (524288, 4096, 1), 0); del buf85  # reuse
    buf91 = buf68; del buf68  # reuse
    buf92 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf94 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf95 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_20(c_void_p(buf90.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del primals_43
    buf96 = buf89; del buf89  # reuse
    # Source Nodes: [query_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf95, reinterpret_tensor(primals_44, (4096, 4096), (1, 4096), 0), out=buf96)
    buf97 = buf67; del buf67  # reuse
    # Source Nodes: [key_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf95, reinterpret_tensor(primals_45, (4096, 4096), (1, 4096), 0), out=buf97)
    buf98 = buf63; del buf63  # reuse
    # Source Nodes: [value_8], Original ATen: [aten.mm]
    extern_kernels.mm(buf95, reinterpret_tensor(primals_46, (4096, 4096), (1, 4096), 0), out=buf98)
    buf99 = reinterpret_tensor(buf45, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf45  # reuse
    buf100 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_21(c_void_p(buf97.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    buf101 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf100, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf99, (16, 256, 128), (256, 1, 4096), 0), out=buf101)
    buf102 = buf79; del buf79  # reuse
    buf103 = reinterpret_tensor(buf101, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf101  # reuse
    buf104 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf105 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_22(c_void_p(buf103.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    buf106 = reinterpret_tensor(buf97, (16, 128, 256), (32768, 256, 1), 0); del buf97  # reuse
    # Source Nodes: [attn_output_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf105, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf98, (16, 128, 256), (256, 4096, 1), 0), out=buf106)
    buf107 = buf96; del buf96  # reuse
    cpp_fused_view_23(c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    buf108 = reinterpret_tensor(buf106, (128, 4096), (4096, 1), 0); del buf106  # reuse
    # Source Nodes: [attn_output_26], Original ATen: [aten.mm]
    extern_kernels.mm(buf107, reinterpret_tensor(primals_47, (4096, 4096), (1, 4096), 0), out=buf108)
    buf109 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_25], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_49, buf95, reinterpret_tensor(primals_48, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf109)
    del primals_49
    buf110 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf111 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_24(c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    buf112 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_51, buf111, reinterpret_tensor(primals_50, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf112)
    del primals_51
    buf113 = buf91; del buf91  # reuse
    buf114 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf116 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf117 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf108.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del primals_53
    buf118 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_25], Original ATen: [aten.mm]
    extern_kernels.mm(buf117, reinterpret_tensor(primals_54, (4096, 4096), (1, 4096), 0), out=buf118)
    buf119 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_25], Original ATen: [aten.mm]
    extern_kernels.mm(buf117, reinterpret_tensor(primals_55, (4096, 4096), (1, 4096), 0), out=buf119)
    buf120 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_10], Original ATen: [aten.mm]
    extern_kernels.mm(buf117, reinterpret_tensor(primals_56, (4096, 4096), (1, 4096), 0), out=buf120)
    buf121 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf122 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_26(c_void_p(buf119.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf122, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf121, (16, 256, 128), (256, 1, 4096), 0), out=buf123)
    buf124 = buf102; del buf102  # reuse
    buf125 = reinterpret_tensor(buf123, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf123  # reuse
    buf126 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf127 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_27(c_void_p(buf125.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    buf128 = reinterpret_tensor(buf119, (16, 128, 256), (32768, 256, 1), 0); del buf119  # reuse
    # Source Nodes: [attn_output_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf127, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf120, (16, 128, 256), (256, 4096, 1), 0), out=buf128)
    buf129 = buf118; del buf118  # reuse
    cpp_fused_view_28(c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = reinterpret_tensor(buf128, (128, 4096), (4096, 1), 0); del buf128  # reuse
    # Source Nodes: [attn_output_32], Original ATen: [aten.mm]
    extern_kernels.mm(buf129, reinterpret_tensor(primals_57, (4096, 4096), (1, 4096), 0), out=buf130)
    buf131 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_31], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_59, buf117, reinterpret_tensor(primals_58, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf131)
    del primals_59
    buf132 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf133 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_29(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    buf134 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_61, buf133, reinterpret_tensor(primals_60, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf134)
    del primals_61
    buf135 = reinterpret_tensor(buf130, (1, 128, 4096), (524288, 4096, 1), 0); del buf130  # reuse
    buf136 = buf113; del buf113  # reuse
    buf137 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf139 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf140 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_30(c_void_p(buf135.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    del primals_63
    buf141 = reinterpret_tensor(buf90, (128, 4096), (4096, 1), 0); del buf90  # reuse
    # Source Nodes: [query_30], Original ATen: [aten.mm]
    extern_kernels.mm(buf140, reinterpret_tensor(primals_64, (4096, 4096), (1, 4096), 0), out=buf141)
    buf142 = buf134; del buf134  # reuse
    # Source Nodes: [key_30], Original ATen: [aten.mm]
    extern_kernels.mm(buf140, reinterpret_tensor(primals_65, (4096, 4096), (1, 4096), 0), out=buf142)
    buf143 = buf112; del buf112  # reuse
    # Source Nodes: [value_12], Original ATen: [aten.mm]
    extern_kernels.mm(buf140, reinterpret_tensor(primals_66, (4096, 4096), (1, 4096), 0), out=buf143)
    buf144 = reinterpret_tensor(buf108, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf108  # reuse
    buf145 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_31(c_void_p(buf142.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf145, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf144, (16, 256, 128), (256, 1, 4096), 0), out=buf146)
    buf147 = buf124; del buf124  # reuse
    buf148 = reinterpret_tensor(buf146, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf146  # reuse
    buf149 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf150 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_32(c_void_p(buf148.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    buf151 = reinterpret_tensor(buf142, (16, 128, 256), (32768, 256, 1), 0); del buf142  # reuse
    # Source Nodes: [attn_output_36], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf150, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf143, (16, 128, 256), (256, 4096, 1), 0), out=buf151)
    buf152 = buf141; del buf141  # reuse
    cpp_fused_view_33(c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    buf153 = reinterpret_tensor(buf151, (128, 4096), (4096, 1), 0); del buf151  # reuse
    # Source Nodes: [attn_output_38], Original ATen: [aten.mm]
    extern_kernels.mm(buf152, reinterpret_tensor(primals_67, (4096, 4096), (1, 4096), 0), out=buf153)
    buf154 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_69, buf140, reinterpret_tensor(primals_68, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf154)
    del primals_69
    buf155 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf156 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_34(c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_71, buf156, reinterpret_tensor(primals_70, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf157)
    del primals_71
    buf158 = buf136; del buf136  # reuse
    buf159 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf161 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf162 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf153.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()))
    del primals_73
    buf163 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_35], Original ATen: [aten.mm]
    extern_kernels.mm(buf162, reinterpret_tensor(primals_74, (4096, 4096), (1, 4096), 0), out=buf163)
    buf164 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_35], Original ATen: [aten.mm]
    extern_kernels.mm(buf162, reinterpret_tensor(primals_75, (4096, 4096), (1, 4096), 0), out=buf164)
    buf165 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_14], Original ATen: [aten.mm]
    extern_kernels.mm(buf162, reinterpret_tensor(primals_76, (4096, 4096), (1, 4096), 0), out=buf165)
    buf166 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf167 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_36(c_void_p(buf164.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_49], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf167, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf166, (16, 256, 128), (256, 1, 4096), 0), out=buf168)
    buf169 = buf147; del buf147  # reuse
    buf170 = reinterpret_tensor(buf168, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf168  # reuse
    buf171 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf172 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_37(c_void_p(buf170.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    buf173 = reinterpret_tensor(buf164, (16, 128, 256), (32768, 256, 1), 0); del buf164  # reuse
    # Source Nodes: [attn_output_42], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf165, (16, 128, 256), (256, 4096, 1), 0), out=buf173)
    buf174 = buf163; del buf163  # reuse
    cpp_fused_view_38(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = reinterpret_tensor(buf173, (128, 4096), (4096, 1), 0); del buf173  # reuse
    # Source Nodes: [attn_output_44], Original ATen: [aten.mm]
    extern_kernels.mm(buf174, reinterpret_tensor(primals_77, (4096, 4096), (1, 4096), 0), out=buf175)
    buf176 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_43], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_79, buf162, reinterpret_tensor(primals_78, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf176)
    del primals_79
    buf177 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf178 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_39(c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    buf179 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_45], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_81, buf178, reinterpret_tensor(primals_80, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf179)
    del primals_81
    buf180 = reinterpret_tensor(buf175, (1, 128, 4096), (524288, 4096, 1), 0); del buf175  # reuse
    buf181 = buf158; del buf158  # reuse
    buf182 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf184 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf185 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_40(c_void_p(buf180.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del primals_83
    buf186 = buf179; del buf179  # reuse
    # Source Nodes: [query_40], Original ATen: [aten.mm]
    extern_kernels.mm(buf185, reinterpret_tensor(primals_84, (4096, 4096), (1, 4096), 0), out=buf186)
    buf187 = buf157; del buf157  # reuse
    # Source Nodes: [key_40], Original ATen: [aten.mm]
    extern_kernels.mm(buf185, reinterpret_tensor(primals_85, (4096, 4096), (1, 4096), 0), out=buf187)
    buf188 = buf153; del buf153  # reuse
    # Source Nodes: [value_16], Original ATen: [aten.mm]
    extern_kernels.mm(buf185, reinterpret_tensor(primals_86, (4096, 4096), (1, 4096), 0), out=buf188)
    buf189 = reinterpret_tensor(buf135, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf135  # reuse
    buf190 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_41(c_void_p(buf187.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_56], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf190, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf189, (16, 256, 128), (256, 1, 4096), 0), out=buf191)
    buf192 = buf169; del buf169  # reuse
    buf193 = reinterpret_tensor(buf191, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf191  # reuse
    buf194 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf195 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_42(c_void_p(buf193.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    buf196 = reinterpret_tensor(buf187, (16, 128, 256), (32768, 256, 1), 0); del buf187  # reuse
    # Source Nodes: [attn_output_48], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf195, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf188, (16, 128, 256), (256, 4096, 1), 0), out=buf196)
    buf197 = buf186; del buf186  # reuse
    cpp_fused_view_43(c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()))
    buf198 = reinterpret_tensor(buf196, (128, 4096), (4096, 1), 0); del buf196  # reuse
    # Source Nodes: [attn_output_50], Original ATen: [aten.mm]
    extern_kernels.mm(buf197, reinterpret_tensor(primals_87, (4096, 4096), (1, 4096), 0), out=buf198)
    buf199 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_49], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_89, buf185, reinterpret_tensor(primals_88, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf199)
    del primals_89
    buf200 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf201 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_44(c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    buf202 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_91, buf201, reinterpret_tensor(primals_90, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf202)
    del primals_91
    buf203 = buf181; del buf181  # reuse
    buf204 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf206 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf207 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_45(c_void_p(buf198.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()))
    del primals_93
    buf208 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_45], Original ATen: [aten.mm]
    extern_kernels.mm(buf207, reinterpret_tensor(primals_94, (4096, 4096), (1, 4096), 0), out=buf208)
    buf209 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_45], Original ATen: [aten.mm]
    extern_kernels.mm(buf207, reinterpret_tensor(primals_95, (4096, 4096), (1, 4096), 0), out=buf209)
    buf210 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_18], Original ATen: [aten.mm]
    extern_kernels.mm(buf207, reinterpret_tensor(primals_96, (4096, 4096), (1, 4096), 0), out=buf210)
    buf211 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf212 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_46(c_void_p(buf209.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()))
    buf213 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_63], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf212, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf211, (16, 256, 128), (256, 1, 4096), 0), out=buf213)
    buf214 = buf192; del buf192  # reuse
    buf215 = reinterpret_tensor(buf213, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf213  # reuse
    buf216 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf217 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_47(c_void_p(buf215.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    buf218 = reinterpret_tensor(buf209, (16, 128, 256), (32768, 256, 1), 0); del buf209  # reuse
    # Source Nodes: [attn_output_54], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf217, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf210, (16, 128, 256), (256, 4096, 1), 0), out=buf218)
    buf219 = buf208; del buf208  # reuse
    cpp_fused_view_48(c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = reinterpret_tensor(buf218, (128, 4096), (4096, 1), 0); del buf218  # reuse
    # Source Nodes: [attn_output_56], Original ATen: [aten.mm]
    extern_kernels.mm(buf219, reinterpret_tensor(primals_97, (4096, 4096), (1, 4096), 0), out=buf220)
    buf221 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_55], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_99, buf207, reinterpret_tensor(primals_98, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf221)
    del primals_99
    buf222 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf223 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_49(c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    buf224 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_57], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_101, buf223, reinterpret_tensor(primals_100, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf224)
    del primals_101
    buf225 = reinterpret_tensor(buf220, (1, 128, 4096), (524288, 4096, 1), 0); del buf220  # reuse
    buf226 = buf203; del buf203  # reuse
    buf227 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf229 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf230 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_50(c_void_p(buf225.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    del primals_103
    buf231 = buf224; del buf224  # reuse
    # Source Nodes: [query_50], Original ATen: [aten.mm]
    extern_kernels.mm(buf230, reinterpret_tensor(primals_104, (4096, 4096), (1, 4096), 0), out=buf231)
    buf232 = buf202; del buf202  # reuse
    # Source Nodes: [key_50], Original ATen: [aten.mm]
    extern_kernels.mm(buf230, reinterpret_tensor(primals_105, (4096, 4096), (1, 4096), 0), out=buf232)
    buf233 = buf198; del buf198  # reuse
    # Source Nodes: [value_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf230, reinterpret_tensor(primals_106, (4096, 4096), (1, 4096), 0), out=buf233)
    buf234 = reinterpret_tensor(buf180, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf180  # reuse
    buf235 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_51(c_void_p(buf232.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    buf236 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_70], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf235, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf234, (16, 256, 128), (256, 1, 4096), 0), out=buf236)
    buf237 = buf214; del buf214  # reuse
    buf238 = reinterpret_tensor(buf236, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf236  # reuse
    buf239 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf240 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_52(c_void_p(buf238.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf232, (16, 128, 256), (32768, 256, 1), 0); del buf232  # reuse
    # Source Nodes: [attn_output_60], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf240, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf233, (16, 128, 256), (256, 4096, 1), 0), out=buf241)
    buf242 = buf231; del buf231  # reuse
    cpp_fused_view_53(c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf241, (128, 4096), (4096, 1), 0); del buf241  # reuse
    # Source Nodes: [attn_output_62], Original ATen: [aten.mm]
    extern_kernels.mm(buf242, reinterpret_tensor(primals_107, (4096, 4096), (1, 4096), 0), out=buf243)
    buf244 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_61], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_109, buf230, reinterpret_tensor(primals_108, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf244)
    del primals_109
    buf245 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf246 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_54(c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()))
    buf247 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_63], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf246, reinterpret_tensor(primals_110, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf247)
    del primals_111
    buf248 = buf226; del buf226  # reuse
    buf249 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf251 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf252 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_55(c_void_p(buf243.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()))
    del primals_113
    buf253 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_55], Original ATen: [aten.mm]
    extern_kernels.mm(buf252, reinterpret_tensor(primals_114, (4096, 4096), (1, 4096), 0), out=buf253)
    buf254 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_55], Original ATen: [aten.mm]
    extern_kernels.mm(buf252, reinterpret_tensor(primals_115, (4096, 4096), (1, 4096), 0), out=buf254)
    buf255 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_22], Original ATen: [aten.mm]
    extern_kernels.mm(buf252, reinterpret_tensor(primals_116, (4096, 4096), (1, 4096), 0), out=buf255)
    buf256 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf257 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_56(c_void_p(buf254.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    buf258 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_77], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf257, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf256, (16, 256, 128), (256, 1, 4096), 0), out=buf258)
    buf259 = buf237; del buf237  # reuse
    buf260 = reinterpret_tensor(buf258, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf258  # reuse
    buf261 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf262 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_57(c_void_p(buf260.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    buf263 = reinterpret_tensor(buf254, (16, 128, 256), (32768, 256, 1), 0); del buf254  # reuse
    # Source Nodes: [attn_output_66], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf262, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf255, (16, 128, 256), (256, 4096, 1), 0), out=buf263)
    buf264 = buf253; del buf253  # reuse
    cpp_fused_view_58(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf263, (128, 4096), (4096, 1), 0); del buf263  # reuse
    # Source Nodes: [attn_output_68], Original ATen: [aten.mm]
    extern_kernels.mm(buf264, reinterpret_tensor(primals_117, (4096, 4096), (1, 4096), 0), out=buf265)
    buf266 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_67], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, buf252, reinterpret_tensor(primals_118, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf266)
    del primals_119
    buf267 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf268 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_59(c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_121, buf268, reinterpret_tensor(primals_120, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf269)
    del primals_121
    buf270 = reinterpret_tensor(buf265, (1, 128, 4096), (524288, 4096, 1), 0); del buf265  # reuse
    buf271 = buf248; del buf248  # reuse
    buf272 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf274 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf275 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_60(c_void_p(buf270.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()))
    del primals_123
    buf276 = buf269; del buf269  # reuse
    # Source Nodes: [query_60], Original ATen: [aten.mm]
    extern_kernels.mm(buf275, reinterpret_tensor(primals_124, (4096, 4096), (1, 4096), 0), out=buf276)
    buf277 = buf247; del buf247  # reuse
    # Source Nodes: [key_60], Original ATen: [aten.mm]
    extern_kernels.mm(buf275, reinterpret_tensor(primals_125, (4096, 4096), (1, 4096), 0), out=buf277)
    buf278 = buf243; del buf243  # reuse
    # Source Nodes: [value_24], Original ATen: [aten.mm]
    extern_kernels.mm(buf275, reinterpret_tensor(primals_126, (4096, 4096), (1, 4096), 0), out=buf278)
    buf279 = reinterpret_tensor(buf225, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf225  # reuse
    buf280 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_61(c_void_p(buf277.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    buf281 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf280, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf279, (16, 256, 128), (256, 1, 4096), 0), out=buf281)
    buf282 = buf259; del buf259  # reuse
    buf283 = reinterpret_tensor(buf281, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf281  # reuse
    buf284 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf285 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_62(c_void_p(buf283.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = reinterpret_tensor(buf277, (16, 128, 256), (32768, 256, 1), 0); del buf277  # reuse
    # Source Nodes: [attn_output_72], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf285, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf278, (16, 128, 256), (256, 4096, 1), 0), out=buf286)
    buf287 = buf276; del buf276  # reuse
    cpp_fused_view_63(c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    buf288 = reinterpret_tensor(buf286, (128, 4096), (4096, 1), 0); del buf286  # reuse
    # Source Nodes: [attn_output_74], Original ATen: [aten.mm]
    extern_kernels.mm(buf287, reinterpret_tensor(primals_127, (4096, 4096), (1, 4096), 0), out=buf288)
    buf289 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_129, buf275, reinterpret_tensor(primals_128, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf289)
    del primals_129
    buf290 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf291 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_64(c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    buf292 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_75], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_131, buf291, reinterpret_tensor(primals_130, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf292)
    del primals_131
    buf293 = buf271; del buf271  # reuse
    buf294 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf296 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf297 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_65(c_void_p(buf288.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()))
    del primals_133
    buf298 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_65], Original ATen: [aten.mm]
    extern_kernels.mm(buf297, reinterpret_tensor(primals_134, (4096, 4096), (1, 4096), 0), out=buf298)
    buf299 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_65], Original ATen: [aten.mm]
    extern_kernels.mm(buf297, reinterpret_tensor(primals_135, (4096, 4096), (1, 4096), 0), out=buf299)
    buf300 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_26], Original ATen: [aten.mm]
    extern_kernels.mm(buf297, reinterpret_tensor(primals_136, (4096, 4096), (1, 4096), 0), out=buf300)
    buf301 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf302 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_66(c_void_p(buf299.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_91], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf302, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf301, (16, 256, 128), (256, 1, 4096), 0), out=buf303)
    buf304 = buf282; del buf282  # reuse
    buf305 = reinterpret_tensor(buf303, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf303  # reuse
    buf306 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf307 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_67(c_void_p(buf305.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    buf308 = reinterpret_tensor(buf299, (16, 128, 256), (32768, 256, 1), 0); del buf299  # reuse
    # Source Nodes: [attn_output_78], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf307, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf300, (16, 128, 256), (256, 4096, 1), 0), out=buf308)
    buf309 = buf298; del buf298  # reuse
    cpp_fused_view_68(c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()))
    buf310 = reinterpret_tensor(buf308, (128, 4096), (4096, 1), 0); del buf308  # reuse
    # Source Nodes: [attn_output_80], Original ATen: [aten.mm]
    extern_kernels.mm(buf309, reinterpret_tensor(primals_137, (4096, 4096), (1, 4096), 0), out=buf310)
    buf311 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_79], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_139, buf297, reinterpret_tensor(primals_138, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf311)
    del primals_139
    buf312 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf313 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_69(c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_141, buf313, reinterpret_tensor(primals_140, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf314)
    del primals_141
    buf315 = reinterpret_tensor(buf310, (1, 128, 4096), (524288, 4096, 1), 0); del buf310  # reuse
    buf316 = buf293; del buf293  # reuse
    buf317 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf319 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf320 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_70(c_void_p(buf315.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del primals_143
    buf321 = buf314; del buf314  # reuse
    # Source Nodes: [query_70], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, reinterpret_tensor(primals_144, (4096, 4096), (1, 4096), 0), out=buf321)
    buf322 = buf292; del buf292  # reuse
    # Source Nodes: [key_70], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, reinterpret_tensor(primals_145, (4096, 4096), (1, 4096), 0), out=buf322)
    buf323 = buf288; del buf288  # reuse
    # Source Nodes: [value_28], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, reinterpret_tensor(primals_146, (4096, 4096), (1, 4096), 0), out=buf323)
    buf324 = reinterpret_tensor(buf270, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf270  # reuse
    buf325 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_71(c_void_p(buf322.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    buf326 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_98], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf325, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf324, (16, 256, 128), (256, 1, 4096), 0), out=buf326)
    buf327 = buf304; del buf304  # reuse
    buf328 = reinterpret_tensor(buf326, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf326  # reuse
    buf329 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf330 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_72(c_void_p(buf328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()))
    buf331 = reinterpret_tensor(buf322, (16, 128, 256), (32768, 256, 1), 0); del buf322  # reuse
    # Source Nodes: [attn_output_84], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf330, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf323, (16, 128, 256), (256, 4096, 1), 0), out=buf331)
    buf332 = buf321; del buf321  # reuse
    cpp_fused_view_73(c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()))
    buf333 = reinterpret_tensor(buf331, (128, 4096), (4096, 1), 0); del buf331  # reuse
    # Source Nodes: [attn_output_86], Original ATen: [aten.mm]
    extern_kernels.mm(buf332, reinterpret_tensor(primals_147, (4096, 4096), (1, 4096), 0), out=buf333)
    buf334 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_85], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf320, reinterpret_tensor(primals_148, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf334)
    del primals_149
    buf335 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf336 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_74(c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_87], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_151, buf336, reinterpret_tensor(primals_150, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf337)
    del primals_151
    buf338 = buf316; del buf316  # reuse
    buf339 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf341 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf342 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_75(c_void_p(buf333.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    del primals_153
    buf343 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_75], Original ATen: [aten.mm]
    extern_kernels.mm(buf342, reinterpret_tensor(primals_154, (4096, 4096), (1, 4096), 0), out=buf343)
    buf344 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_75], Original ATen: [aten.mm]
    extern_kernels.mm(buf342, reinterpret_tensor(primals_155, (4096, 4096), (1, 4096), 0), out=buf344)
    buf345 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_30], Original ATen: [aten.mm]
    extern_kernels.mm(buf342, reinterpret_tensor(primals_156, (4096, 4096), (1, 4096), 0), out=buf345)
    buf346 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf347 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_76(c_void_p(buf344.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()))
    buf348 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_105], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf347, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf346, (16, 256, 128), (256, 1, 4096), 0), out=buf348)
    buf349 = buf327; del buf327  # reuse
    buf350 = reinterpret_tensor(buf348, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf348  # reuse
    buf351 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf352 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_77(c_void_p(buf350.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    buf353 = reinterpret_tensor(buf344, (16, 128, 256), (32768, 256, 1), 0); del buf344  # reuse
    # Source Nodes: [attn_output_90], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf352, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf345, (16, 128, 256), (256, 4096, 1), 0), out=buf353)
    buf354 = buf343; del buf343  # reuse
    cpp_fused_view_78(c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    buf355 = reinterpret_tensor(buf353, (128, 4096), (4096, 1), 0); del buf353  # reuse
    # Source Nodes: [attn_output_92], Original ATen: [aten.mm]
    extern_kernels.mm(buf354, reinterpret_tensor(primals_157, (4096, 4096), (1, 4096), 0), out=buf355)
    buf356 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_159, buf342, reinterpret_tensor(primals_158, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf356)
    del primals_159
    buf357 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf358 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_79(c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    buf359 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_93], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_161, buf358, reinterpret_tensor(primals_160, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf359)
    del primals_161
    buf360 = reinterpret_tensor(buf355, (1, 128, 4096), (524288, 4096, 1), 0); del buf355  # reuse
    buf361 = buf338; del buf338  # reuse
    buf362 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf364 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf365 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_80(c_void_p(buf360.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()))
    del primals_163
    buf366 = buf359; del buf359  # reuse
    # Source Nodes: [query_80], Original ATen: [aten.mm]
    extern_kernels.mm(buf365, reinterpret_tensor(primals_164, (4096, 4096), (1, 4096), 0), out=buf366)
    buf367 = buf337; del buf337  # reuse
    # Source Nodes: [key_80], Original ATen: [aten.mm]
    extern_kernels.mm(buf365, reinterpret_tensor(primals_165, (4096, 4096), (1, 4096), 0), out=buf367)
    buf368 = buf333; del buf333  # reuse
    # Source Nodes: [value_32], Original ATen: [aten.mm]
    extern_kernels.mm(buf365, reinterpret_tensor(primals_166, (4096, 4096), (1, 4096), 0), out=buf368)
    buf369 = reinterpret_tensor(buf315, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf315  # reuse
    buf370 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_81(c_void_p(buf367.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    buf371 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_112], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf370, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf369, (16, 256, 128), (256, 1, 4096), 0), out=buf371)
    buf372 = buf349; del buf349  # reuse
    buf373 = reinterpret_tensor(buf371, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf371  # reuse
    buf374 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf375 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_82(c_void_p(buf373.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    buf376 = reinterpret_tensor(buf367, (16, 128, 256), (32768, 256, 1), 0); del buf367  # reuse
    # Source Nodes: [attn_output_96], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf375, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf368, (16, 128, 256), (256, 4096, 1), 0), out=buf376)
    buf377 = buf366; del buf366  # reuse
    cpp_fused_view_83(c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    buf378 = reinterpret_tensor(buf376, (128, 4096), (4096, 1), 0); del buf376  # reuse
    # Source Nodes: [attn_output_98], Original ATen: [aten.mm]
    extern_kernels.mm(buf377, reinterpret_tensor(primals_167, (4096, 4096), (1, 4096), 0), out=buf378)
    buf379 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_97], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_169, buf365, reinterpret_tensor(primals_168, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf379)
    del primals_169
    buf380 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf381 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_84(c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()))
    buf382 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_171, buf381, reinterpret_tensor(primals_170, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf382)
    del primals_171
    buf383 = buf361; del buf361  # reuse
    buf384 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf386 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf387 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_85(c_void_p(buf378.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()))
    del primals_173
    buf388 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_85], Original ATen: [aten.mm]
    extern_kernels.mm(buf387, reinterpret_tensor(primals_174, (4096, 4096), (1, 4096), 0), out=buf388)
    buf389 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_85], Original ATen: [aten.mm]
    extern_kernels.mm(buf387, reinterpret_tensor(primals_175, (4096, 4096), (1, 4096), 0), out=buf389)
    buf390 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_34], Original ATen: [aten.mm]
    extern_kernels.mm(buf387, reinterpret_tensor(primals_176, (4096, 4096), (1, 4096), 0), out=buf390)
    buf391 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf392 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_86(c_void_p(buf389.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()))
    buf393 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_119], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf392, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf391, (16, 256, 128), (256, 1, 4096), 0), out=buf393)
    buf394 = buf372; del buf372  # reuse
    buf395 = reinterpret_tensor(buf393, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf393  # reuse
    buf396 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf397 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_87(c_void_p(buf395.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()))
    buf398 = reinterpret_tensor(buf389, (16, 128, 256), (32768, 256, 1), 0); del buf389  # reuse
    # Source Nodes: [attn_output_102], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf397, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf390, (16, 128, 256), (256, 4096, 1), 0), out=buf398)
    buf399 = buf388; del buf388  # reuse
    cpp_fused_view_88(c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    buf400 = reinterpret_tensor(buf398, (128, 4096), (4096, 1), 0); del buf398  # reuse
    # Source Nodes: [attn_output_104], Original ATen: [aten.mm]
    extern_kernels.mm(buf399, reinterpret_tensor(primals_177, (4096, 4096), (1, 4096), 0), out=buf400)
    buf401 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_103], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_179, buf387, reinterpret_tensor(primals_178, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf401)
    del primals_179
    buf402 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf403 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_89(c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    buf404 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_105], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_181, buf403, reinterpret_tensor(primals_180, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf404)
    del primals_181
    buf405 = reinterpret_tensor(buf400, (1, 128, 4096), (524288, 4096, 1), 0); del buf400  # reuse
    buf406 = buf383; del buf383  # reuse
    buf407 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf409 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf410 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_90(c_void_p(buf405.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()))
    del primals_183
    buf411 = buf404; del buf404  # reuse
    # Source Nodes: [query_90], Original ATen: [aten.mm]
    extern_kernels.mm(buf410, reinterpret_tensor(primals_184, (4096, 4096), (1, 4096), 0), out=buf411)
    buf412 = buf382; del buf382  # reuse
    # Source Nodes: [key_90], Original ATen: [aten.mm]
    extern_kernels.mm(buf410, reinterpret_tensor(primals_185, (4096, 4096), (1, 4096), 0), out=buf412)
    buf413 = buf378; del buf378  # reuse
    # Source Nodes: [value_36], Original ATen: [aten.mm]
    extern_kernels.mm(buf410, reinterpret_tensor(primals_186, (4096, 4096), (1, 4096), 0), out=buf413)
    buf414 = reinterpret_tensor(buf360, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf360  # reuse
    buf415 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_91(c_void_p(buf412.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()))
    buf416 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf415, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf414, (16, 256, 128), (256, 1, 4096), 0), out=buf416)
    buf417 = buf394; del buf394  # reuse
    buf418 = reinterpret_tensor(buf416, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf416  # reuse
    buf419 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf420 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_92(c_void_p(buf418.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()))
    buf421 = reinterpret_tensor(buf412, (16, 128, 256), (32768, 256, 1), 0); del buf412  # reuse
    # Source Nodes: [attn_output_108], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf420, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf413, (16, 128, 256), (256, 4096, 1), 0), out=buf421)
    buf422 = buf411; del buf411  # reuse
    cpp_fused_view_93(c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()))
    buf423 = reinterpret_tensor(buf421, (128, 4096), (4096, 1), 0); del buf421  # reuse
    # Source Nodes: [attn_output_110], Original ATen: [aten.mm]
    extern_kernels.mm(buf422, reinterpret_tensor(primals_187, (4096, 4096), (1, 4096), 0), out=buf423)
    buf424 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_189, buf410, reinterpret_tensor(primals_188, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf424)
    del primals_189
    buf425 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf426 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_94(c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()))
    buf427 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_111], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_191, buf426, reinterpret_tensor(primals_190, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf427)
    del primals_191
    buf428 = buf406; del buf406  # reuse
    buf429 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf431 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf432 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_95(c_void_p(buf423.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(primals_192.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()))
    del primals_193
    buf433 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_95], Original ATen: [aten.mm]
    extern_kernels.mm(buf432, reinterpret_tensor(primals_194, (4096, 4096), (1, 4096), 0), out=buf433)
    buf434 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_95], Original ATen: [aten.mm]
    extern_kernels.mm(buf432, reinterpret_tensor(primals_195, (4096, 4096), (1, 4096), 0), out=buf434)
    buf435 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_38], Original ATen: [aten.mm]
    extern_kernels.mm(buf432, reinterpret_tensor(primals_196, (4096, 4096), (1, 4096), 0), out=buf435)
    buf436 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf437 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_96(c_void_p(buf434.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    buf438 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_133], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf437, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf436, (16, 256, 128), (256, 1, 4096), 0), out=buf438)
    buf439 = buf417; del buf417  # reuse
    buf440 = reinterpret_tensor(buf438, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf438  # reuse
    buf441 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf442 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_97(c_void_p(buf440.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()))
    buf443 = reinterpret_tensor(buf434, (16, 128, 256), (32768, 256, 1), 0); del buf434  # reuse
    # Source Nodes: [attn_output_114], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf442, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf435, (16, 128, 256), (256, 4096, 1), 0), out=buf443)
    buf444 = buf433; del buf433  # reuse
    cpp_fused_view_98(c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()))
    buf445 = reinterpret_tensor(buf443, (128, 4096), (4096, 1), 0); del buf443  # reuse
    # Source Nodes: [attn_output_116], Original ATen: [aten.mm]
    extern_kernels.mm(buf444, reinterpret_tensor(primals_197, (4096, 4096), (1, 4096), 0), out=buf445)
    buf446 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_199, buf432, reinterpret_tensor(primals_198, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf446)
    del primals_199
    buf447 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf448 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_99(c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_201, buf448, reinterpret_tensor(primals_200, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf449)
    del primals_201
    buf450 = reinterpret_tensor(buf445, (1, 128, 4096), (524288, 4096, 1), 0); del buf445  # reuse
    buf451 = buf428; del buf428  # reuse
    buf452 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf454 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf455 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_100(c_void_p(buf450.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()))
    del primals_203
    buf456 = buf449; del buf449  # reuse
    # Source Nodes: [query_100], Original ATen: [aten.mm]
    extern_kernels.mm(buf455, reinterpret_tensor(primals_204, (4096, 4096), (1, 4096), 0), out=buf456)
    buf457 = buf427; del buf427  # reuse
    # Source Nodes: [key_100], Original ATen: [aten.mm]
    extern_kernels.mm(buf455, reinterpret_tensor(primals_205, (4096, 4096), (1, 4096), 0), out=buf457)
    buf458 = buf423; del buf423  # reuse
    # Source Nodes: [value_40], Original ATen: [aten.mm]
    extern_kernels.mm(buf455, reinterpret_tensor(primals_206, (4096, 4096), (1, 4096), 0), out=buf458)
    buf459 = reinterpret_tensor(buf405, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf405  # reuse
    buf460 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_101(c_void_p(buf457.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()))
    buf461 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_140], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf460, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf459, (16, 256, 128), (256, 1, 4096), 0), out=buf461)
    buf462 = buf439; del buf439  # reuse
    buf463 = reinterpret_tensor(buf461, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf461  # reuse
    buf464 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf465 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_102(c_void_p(buf463.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()))
    buf466 = reinterpret_tensor(buf457, (16, 128, 256), (32768, 256, 1), 0); del buf457  # reuse
    # Source Nodes: [attn_output_120], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf465, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf458, (16, 128, 256), (256, 4096, 1), 0), out=buf466)
    buf467 = buf456; del buf456  # reuse
    cpp_fused_view_103(c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()))
    buf468 = reinterpret_tensor(buf466, (128, 4096), (4096, 1), 0); del buf466  # reuse
    # Source Nodes: [attn_output_122], Original ATen: [aten.mm]
    extern_kernels.mm(buf467, reinterpret_tensor(primals_207, (4096, 4096), (1, 4096), 0), out=buf468)
    buf469 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_121], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_209, buf455, reinterpret_tensor(primals_208, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf469)
    del primals_209
    buf470 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf471 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_104(c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()))
    buf472 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_123], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_211, buf471, reinterpret_tensor(primals_210, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf472)
    del primals_211
    buf473 = buf451; del buf451  # reuse
    buf474 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf476 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf477 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_105(c_void_p(buf468.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()))
    del primals_213
    buf478 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_105], Original ATen: [aten.mm]
    extern_kernels.mm(buf477, reinterpret_tensor(primals_214, (4096, 4096), (1, 4096), 0), out=buf478)
    buf479 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_105], Original ATen: [aten.mm]
    extern_kernels.mm(buf477, reinterpret_tensor(primals_215, (4096, 4096), (1, 4096), 0), out=buf479)
    buf480 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_42], Original ATen: [aten.mm]
    extern_kernels.mm(buf477, reinterpret_tensor(primals_216, (4096, 4096), (1, 4096), 0), out=buf480)
    buf481 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf482 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_106(c_void_p(buf479.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf482.data_ptr()))
    buf483 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_147], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf482, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf481, (16, 256, 128), (256, 1, 4096), 0), out=buf483)
    buf484 = buf462; del buf462  # reuse
    buf485 = reinterpret_tensor(buf483, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf483  # reuse
    buf486 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf487 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_107(c_void_p(buf485.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()))
    buf488 = reinterpret_tensor(buf479, (16, 128, 256), (32768, 256, 1), 0); del buf479  # reuse
    # Source Nodes: [attn_output_126], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf487, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf480, (16, 128, 256), (256, 4096, 1), 0), out=buf488)
    buf489 = buf478; del buf478  # reuse
    cpp_fused_view_108(c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()))
    buf490 = reinterpret_tensor(buf488, (128, 4096), (4096, 1), 0); del buf488  # reuse
    # Source Nodes: [attn_output_128], Original ATen: [aten.mm]
    extern_kernels.mm(buf489, reinterpret_tensor(primals_217, (4096, 4096), (1, 4096), 0), out=buf490)
    buf491 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_127], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_219, buf477, reinterpret_tensor(primals_218, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf491)
    del primals_219
    buf492 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf493 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_109(c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()))
    buf494 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_221, buf493, reinterpret_tensor(primals_220, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf494)
    del primals_221
    buf495 = reinterpret_tensor(buf490, (1, 128, 4096), (524288, 4096, 1), 0); del buf490  # reuse
    buf496 = buf473; del buf473  # reuse
    buf497 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf499 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf500 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_110(c_void_p(buf495.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()))
    del primals_223
    buf501 = buf494; del buf494  # reuse
    # Source Nodes: [query_110], Original ATen: [aten.mm]
    extern_kernels.mm(buf500, reinterpret_tensor(primals_224, (4096, 4096), (1, 4096), 0), out=buf501)
    buf502 = buf472; del buf472  # reuse
    # Source Nodes: [key_110], Original ATen: [aten.mm]
    extern_kernels.mm(buf500, reinterpret_tensor(primals_225, (4096, 4096), (1, 4096), 0), out=buf502)
    buf503 = buf468; del buf468  # reuse
    # Source Nodes: [value_44], Original ATen: [aten.mm]
    extern_kernels.mm(buf500, reinterpret_tensor(primals_226, (4096, 4096), (1, 4096), 0), out=buf503)
    buf504 = reinterpret_tensor(buf450, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf450  # reuse
    buf505 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_111(c_void_p(buf502.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf505.data_ptr()))
    buf506 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_154], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf505, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf504, (16, 256, 128), (256, 1, 4096), 0), out=buf506)
    buf507 = buf484; del buf484  # reuse
    buf508 = reinterpret_tensor(buf506, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf506  # reuse
    buf509 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf510 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_112(c_void_p(buf508.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()))
    buf511 = reinterpret_tensor(buf502, (16, 128, 256), (32768, 256, 1), 0); del buf502  # reuse
    # Source Nodes: [attn_output_132], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf510, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf503, (16, 128, 256), (256, 4096, 1), 0), out=buf511)
    buf512 = buf501; del buf501  # reuse
    cpp_fused_view_113(c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()))
    buf513 = reinterpret_tensor(buf511, (128, 4096), (4096, 1), 0); del buf511  # reuse
    # Source Nodes: [attn_output_134], Original ATen: [aten.mm]
    extern_kernels.mm(buf512, reinterpret_tensor(primals_227, (4096, 4096), (1, 4096), 0), out=buf513)
    buf514 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_229, buf500, reinterpret_tensor(primals_228, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf514)
    del primals_229
    buf515 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf516 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_114(c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()))
    buf517 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_135], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_231, buf516, reinterpret_tensor(primals_230, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf517)
    del primals_231
    buf518 = buf496; del buf496  # reuse
    buf519 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf521 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf522 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_115(c_void_p(buf513.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()))
    del primals_233
    buf523 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_115], Original ATen: [aten.mm]
    extern_kernels.mm(buf522, reinterpret_tensor(primals_234, (4096, 4096), (1, 4096), 0), out=buf523)
    buf524 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_115], Original ATen: [aten.mm]
    extern_kernels.mm(buf522, reinterpret_tensor(primals_235, (4096, 4096), (1, 4096), 0), out=buf524)
    buf525 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_46], Original ATen: [aten.mm]
    extern_kernels.mm(buf522, reinterpret_tensor(primals_236, (4096, 4096), (1, 4096), 0), out=buf525)
    buf526 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf527 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_116(c_void_p(buf524.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()))
    buf528 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_161], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf527, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf526, (16, 256, 128), (256, 1, 4096), 0), out=buf528)
    buf529 = buf507; del buf507  # reuse
    buf530 = reinterpret_tensor(buf528, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf528  # reuse
    buf531 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf532 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_117(c_void_p(buf530.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    buf533 = reinterpret_tensor(buf524, (16, 128, 256), (32768, 256, 1), 0); del buf524  # reuse
    # Source Nodes: [attn_output_138], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf532, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf525, (16, 128, 256), (256, 4096, 1), 0), out=buf533)
    buf534 = buf523; del buf523  # reuse
    cpp_fused_view_118(c_void_p(buf533.data_ptr()), c_void_p(buf534.data_ptr()))
    buf535 = reinterpret_tensor(buf533, (128, 4096), (4096, 1), 0); del buf533  # reuse
    # Source Nodes: [attn_output_140], Original ATen: [aten.mm]
    extern_kernels.mm(buf534, reinterpret_tensor(primals_237, (4096, 4096), (1, 4096), 0), out=buf535)
    buf536 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_239, buf522, reinterpret_tensor(primals_238, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf536)
    del primals_239
    buf537 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf538 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_119(c_void_p(buf536.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf538.data_ptr()))
    buf539 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_241, buf538, reinterpret_tensor(primals_240, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf539)
    del primals_241
    buf540 = reinterpret_tensor(buf535, (1, 128, 4096), (524288, 4096, 1), 0); del buf535  # reuse
    buf541 = buf518; del buf518  # reuse
    buf542 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf544 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf545 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_120(c_void_p(buf540.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()))
    del primals_243
    buf546 = buf539; del buf539  # reuse
    # Source Nodes: [query_120], Original ATen: [aten.mm]
    extern_kernels.mm(buf545, reinterpret_tensor(primals_244, (4096, 4096), (1, 4096), 0), out=buf546)
    buf547 = buf517; del buf517  # reuse
    # Source Nodes: [key_120], Original ATen: [aten.mm]
    extern_kernels.mm(buf545, reinterpret_tensor(primals_245, (4096, 4096), (1, 4096), 0), out=buf547)
    buf548 = buf513; del buf513  # reuse
    # Source Nodes: [value_48], Original ATen: [aten.mm]
    extern_kernels.mm(buf545, reinterpret_tensor(primals_246, (4096, 4096), (1, 4096), 0), out=buf548)
    buf549 = reinterpret_tensor(buf495, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf495  # reuse
    buf550 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_121(c_void_p(buf547.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf549.data_ptr()), c_void_p(buf550.data_ptr()))
    buf551 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_168], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf550, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf549, (16, 256, 128), (256, 1, 4096), 0), out=buf551)
    buf552 = buf529; del buf529  # reuse
    buf553 = reinterpret_tensor(buf551, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf551  # reuse
    buf554 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf555 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_122(c_void_p(buf553.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()))
    buf556 = reinterpret_tensor(buf547, (16, 128, 256), (32768, 256, 1), 0); del buf547  # reuse
    # Source Nodes: [attn_output_144], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf555, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf548, (16, 128, 256), (256, 4096, 1), 0), out=buf556)
    buf557 = buf546; del buf546  # reuse
    cpp_fused_view_123(c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()))
    buf558 = reinterpret_tensor(buf556, (128, 4096), (4096, 1), 0); del buf556  # reuse
    # Source Nodes: [attn_output_146], Original ATen: [aten.mm]
    extern_kernels.mm(buf557, reinterpret_tensor(primals_247, (4096, 4096), (1, 4096), 0), out=buf558)
    buf559 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_145], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_249, buf545, reinterpret_tensor(primals_248, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf559)
    del primals_249
    buf560 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf561 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_124(c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()))
    buf562 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_147], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_251, buf561, reinterpret_tensor(primals_250, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf562)
    del primals_251
    buf563 = buf541; del buf541  # reuse
    buf564 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf566 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf567 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_125(c_void_p(buf558.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf566.data_ptr()), c_void_p(buf567.data_ptr()))
    del primals_253
    buf568 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_125], Original ATen: [aten.mm]
    extern_kernels.mm(buf567, reinterpret_tensor(primals_254, (4096, 4096), (1, 4096), 0), out=buf568)
    buf569 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_125], Original ATen: [aten.mm]
    extern_kernels.mm(buf567, reinterpret_tensor(primals_255, (4096, 4096), (1, 4096), 0), out=buf569)
    buf570 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_50], Original ATen: [aten.mm]
    extern_kernels.mm(buf567, reinterpret_tensor(primals_256, (4096, 4096), (1, 4096), 0), out=buf570)
    buf571 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf572 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_126(c_void_p(buf569.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()))
    buf573 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_175], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf572, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf571, (16, 256, 128), (256, 1, 4096), 0), out=buf573)
    buf574 = buf552; del buf552  # reuse
    buf575 = reinterpret_tensor(buf573, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf573  # reuse
    buf576 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf577 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_div_lift_fresh_where_127(c_void_p(buf575.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()))
    buf578 = reinterpret_tensor(buf569, (16, 128, 256), (32768, 256, 1), 0); del buf569  # reuse
    # Source Nodes: [attn_output_150], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf577, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf570, (16, 128, 256), (256, 4096, 1), 0), out=buf578)
    buf579 = buf568; del buf568  # reuse
    cpp_fused_view_128(c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()))
    buf580 = reinterpret_tensor(buf578, (128, 4096), (4096, 1), 0); del buf578  # reuse
    # Source Nodes: [attn_output_152], Original ATen: [aten.mm]
    extern_kernels.mm(buf579, reinterpret_tensor(primals_257, (4096, 4096), (1, 4096), 0), out=buf580)
    buf581 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_259, buf567, reinterpret_tensor(primals_258, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf581)
    del primals_259
    buf582 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf583 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_129(c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()))
    buf584 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_261, buf583, reinterpret_tensor(primals_260, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf584)
    del primals_261
    buf585 = reinterpret_tensor(buf580, (1, 128, 4096), (524288, 4096, 1), 0); del buf580  # reuse
    buf586 = buf563; del buf563  # reuse
    buf587 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf589 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf590 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_130(c_void_p(buf585.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf590.data_ptr()))
    del primals_263
    buf591 = buf584; del buf584  # reuse
    # Source Nodes: [query_130], Original ATen: [aten.mm]
    extern_kernels.mm(buf590, reinterpret_tensor(primals_264, (4096, 4096), (1, 4096), 0), out=buf591)
    buf592 = buf562; del buf562  # reuse
    # Source Nodes: [key_130], Original ATen: [aten.mm]
    extern_kernels.mm(buf590, reinterpret_tensor(primals_265, (4096, 4096), (1, 4096), 0), out=buf592)
    buf593 = buf558; del buf558  # reuse
    # Source Nodes: [value_52], Original ATen: [aten.mm]
    extern_kernels.mm(buf590, reinterpret_tensor(primals_266, (4096, 4096), (1, 4096), 0), out=buf593)
    buf594 = reinterpret_tensor(buf540, (1, 16, 128, 256), (524288, 256, 4096, 1), 0); del buf540  # reuse
    buf595 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_131(c_void_p(buf592.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf595.data_ptr()))
    buf596 = empty((16, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_weights_182], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf595, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf594, (16, 256, 128), (256, 1, 4096), 0), out=buf596)
    buf597 = buf574; del buf574  # reuse
    buf598 = reinterpret_tensor(buf596, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf596  # reuse
    buf599 = empty_strided((1, 16, 128, 1), (2048, 128, 1, 2048), device='cpu', dtype=torch.float32)
    buf600 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    buf646 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_detach_div_lift_fresh_where_132(c_void_p(buf598.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf646.data_ptr()))
    buf601 = reinterpret_tensor(buf592, (16, 128, 256), (32768, 256, 1), 0); del buf592  # reuse
    # Source Nodes: [attn_output_156], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf600, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf593, (16, 128, 256), (256, 4096, 1), 0), out=buf601)
    buf602 = buf591; del buf591  # reuse
    cpp_fused_view_133(c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()))
    buf603 = reinterpret_tensor(buf601, (128, 4096), (4096, 1), 0); del buf601  # reuse
    # Source Nodes: [attn_output_158], Original ATen: [aten.mm]
    extern_kernels.mm(buf602, reinterpret_tensor(primals_267, (4096, 4096), (1, 4096), 0), out=buf603)
    buf604 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_269, buf590, reinterpret_tensor(primals_268, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf604)
    del primals_269
    buf605 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf606 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_134(c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()))
    buf607 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_271, buf606, reinterpret_tensor(primals_270, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf607)
    del primals_271
    buf608 = buf586; del buf586  # reuse
    buf609 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf611 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf612 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_135(c_void_p(buf603.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()))
    del primals_273
    buf613 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [query_135], Original ATen: [aten.mm]
    extern_kernels.mm(buf612, reinterpret_tensor(primals_274, (4096, 4096), (1, 4096), 0), out=buf613)
    buf614 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [key_135], Original ATen: [aten.mm]
    extern_kernels.mm(buf612, reinterpret_tensor(primals_275, (4096, 4096), (1, 4096), 0), out=buf614)
    buf615 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [value_54], Original ATen: [aten.mm]
    extern_kernels.mm(buf612, reinterpret_tensor(primals_276, (4096, 4096), (1, 4096), 0), out=buf615)
    buf616 = empty_strided((1, 16, 128, 256), (524288, 256, 4096, 1), device='cpu', dtype=torch.float32)
    buf617 = empty((1, 128, 16, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_permute_136(c_void_p(buf614.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()))
    buf618 = reinterpret_tensor(buf598, (16, 128, 128), (16384, 128, 1), 0); del buf598  # reuse
    # Source Nodes: [attn_weights_189], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf617, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(buf616, (16, 256, 128), (256, 1, 4096), 0), out=buf618)
    buf619 = buf599; del buf599  # reuse
    buf620 = reinterpret_tensor(buf618, (1, 16, 128, 128), (262144, 16384, 128, 1), 0); del buf618  # reuse
    buf621 = buf597; del buf597  # reuse
    buf622 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    buf644 = empty((1, 16, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_detach_div_lift_fresh_where_137(c_void_p(buf620.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf644.data_ptr()))
    del buf619
    del buf620
    del buf621
    buf623 = reinterpret_tensor(buf614, (16, 128, 256), (32768, 256, 1), 0); del buf614  # reuse
    # Source Nodes: [attn_output_162], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf622, (16, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf615, (16, 128, 256), (256, 4096, 1), 0), out=buf623)
    buf624 = buf613; del buf613  # reuse
    cpp_fused_view_138(c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()))
    buf625 = reinterpret_tensor(buf623, (128, 4096), (4096, 1), 0); del buf623  # reuse
    # Source Nodes: [attn_output_164], Original ATen: [aten.mm]
    extern_kernels.mm(buf624, reinterpret_tensor(primals_277, (4096, 4096), (1, 4096), 0), out=buf625)
    buf626 = empty((128, 16384), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_279, buf612, reinterpret_tensor(primals_278, (4096, 16384), (1, 4096), 0), alpha=1, beta=1, out=buf626)
    del primals_279
    buf627 = empty((1, 128, 16384), device='cpu', dtype=torch.float32)
    buf628 = empty((128, 16384), device='cpu', dtype=torch.float32)
    cpp_fused_add_mul_pow_tanh_view_139(c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf628.data_ptr()))
    buf629 = empty((128, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [hidden_states_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_281, buf628, reinterpret_tensor(primals_280, (16384, 4096), (1, 16384), 0), alpha=1, beta=1, out=buf629)
    del primals_281
    buf630 = reinterpret_tensor(buf625, (1, 128, 4096), (524288, 4096, 1), 0); del buf625  # reuse
    buf631 = buf608; del buf608  # reuse
    buf632 = empty_strided((1, 128, 1), (128, 1, 128), device='cpu', dtype=torch.float32)
    buf634 = empty((1, 128, 4096), device='cpu', dtype=torch.float32)
    buf635 = empty((128, 4096), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_140(c_void_p(buf630.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf635.data_ptr()))
    del buf585
    del buf603
    del buf607
    del buf629
    del buf630
    del buf631
    del primals_283
    buf636 = empty((128, 50400), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___lm_head], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_285, buf635, reinterpret_tensor(primals_284, (4096, 50400), (1, 4096), 0), alpha=1, beta=1, out=buf636)
    del primals_285
    buf637 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.float32)
    buf638 = empty_strided((127, 1), (1, 127), device='cpu', dtype=torch.float32)
    buf639 = empty((127, 50400), device='cpu', dtype=torch.float32)
    buf640 = empty((), device='cpu', dtype=torch.int64)
    buf642 = empty((), device='cpu', dtype=torch.float32)
    buf641 = empty((), device='cpu', dtype=torch.float32)
    buf699 = buf642; del buf642  # reuse
    buf643 = reinterpret_tensor(buf632, (1, 128, 1), (128, 1, 1), 0); del buf632  # reuse
    buf645 = reinterpret_tensor(buf609, (1, 128, 1), (128, 1, 1), 0); del buf609  # reuse
    buf647 = reinterpret_tensor(buf587, (1, 128, 1), (128, 1, 1), 0); del buf587  # reuse
    buf648 = buf575; del buf575  # reuse
    buf649 = reinterpret_tensor(buf564, (1, 128, 1), (128, 1, 1), 0); del buf564  # reuse
    buf650 = buf553; del buf553  # reuse
    buf651 = reinterpret_tensor(buf542, (1, 128, 1), (128, 1, 1), 0); del buf542  # reuse
    buf652 = buf530; del buf530  # reuse
    buf653 = reinterpret_tensor(buf519, (1, 128, 1), (128, 1, 1), 0); del buf519  # reuse
    buf654 = buf508; del buf508  # reuse
    buf655 = reinterpret_tensor(buf497, (1, 128, 1), (128, 1, 1), 0); del buf497  # reuse
    buf656 = buf485; del buf485  # reuse
    buf657 = reinterpret_tensor(buf474, (1, 128, 1), (128, 1, 1), 0); del buf474  # reuse
    buf658 = buf463; del buf463  # reuse
    buf659 = reinterpret_tensor(buf452, (1, 128, 1), (128, 1, 1), 0); del buf452  # reuse
    buf660 = buf440; del buf440  # reuse
    buf661 = reinterpret_tensor(buf429, (1, 128, 1), (128, 1, 1), 0); del buf429  # reuse
    buf662 = buf418; del buf418  # reuse
    buf663 = reinterpret_tensor(buf407, (1, 128, 1), (128, 1, 1), 0); del buf407  # reuse
    buf664 = buf395; del buf395  # reuse
    buf665 = reinterpret_tensor(buf384, (1, 128, 1), (128, 1, 1), 0); del buf384  # reuse
    buf666 = buf373; del buf373  # reuse
    buf667 = reinterpret_tensor(buf362, (1, 128, 1), (128, 1, 1), 0); del buf362  # reuse
    buf668 = buf350; del buf350  # reuse
    buf669 = reinterpret_tensor(buf339, (1, 128, 1), (128, 1, 1), 0); del buf339  # reuse
    buf670 = buf328; del buf328  # reuse
    buf671 = reinterpret_tensor(buf317, (1, 128, 1), (128, 1, 1), 0); del buf317  # reuse
    buf672 = buf305; del buf305  # reuse
    buf673 = reinterpret_tensor(buf294, (1, 128, 1), (128, 1, 1), 0); del buf294  # reuse
    buf674 = buf283; del buf283  # reuse
    buf675 = reinterpret_tensor(buf272, (1, 128, 1), (128, 1, 1), 0); del buf272  # reuse
    buf676 = buf260; del buf260  # reuse
    buf677 = reinterpret_tensor(buf249, (1, 128, 1), (128, 1, 1), 0); del buf249  # reuse
    buf678 = buf238; del buf238  # reuse
    buf679 = reinterpret_tensor(buf227, (1, 128, 1), (128, 1, 1), 0); del buf227  # reuse
    buf680 = buf215; del buf215  # reuse
    buf681 = reinterpret_tensor(buf204, (1, 128, 1), (128, 1, 1), 0); del buf204  # reuse
    buf682 = buf193; del buf193  # reuse
    buf683 = reinterpret_tensor(buf182, (1, 128, 1), (128, 1, 1), 0); del buf182  # reuse
    buf684 = buf170; del buf170  # reuse
    buf685 = reinterpret_tensor(buf159, (1, 128, 1), (128, 1, 1), 0); del buf159  # reuse
    buf686 = buf148; del buf148  # reuse
    buf687 = reinterpret_tensor(buf137, (1, 128, 1), (128, 1, 1), 0); del buf137  # reuse
    buf688 = buf125; del buf125  # reuse
    buf689 = reinterpret_tensor(buf114, (1, 128, 1), (128, 1, 1), 0); del buf114  # reuse
    buf690 = buf103; del buf103  # reuse
    buf691 = reinterpret_tensor(buf92, (1, 128, 1), (128, 1, 1), 0); del buf92  # reuse
    buf692 = buf80; del buf80  # reuse
    buf693 = reinterpret_tensor(buf69, (1, 128, 1), (128, 1, 1), 0); del buf69  # reuse
    buf694 = buf58; del buf58  # reuse
    buf695 = reinterpret_tensor(buf47, (1, 128, 1), (128, 1, 1), 0); del buf47  # reuse
    buf696 = buf35; del buf35  # reuse
    buf697 = reinterpret_tensor(buf24, (1, 128, 1), (128, 1, 1), 0); del buf24  # reuse
    buf698 = buf13; del buf13  # reuse
    buf700 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf701 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf702 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf703 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf704 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf705 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf706 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf707 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf708 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf709 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf710 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf711 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf712 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf713 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf714 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf715 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf716 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf717 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf718 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf719 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf720 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf721 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf722 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf723 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf724 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf725 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf726 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    buf727 = empty((1, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused__log_softmax__softmax_add_detach_gather_native_layer_norm_native_layer_norm_backward_nll_loss_forward_141(c_void_p(buf699.data_ptr()), c_void_p(buf643.data_ptr()), c_void_p(buf645.data_ptr()), c_void_p(buf647.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf649.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf651.data_ptr()), c_void_p(buf652.data_ptr()), c_void_p(buf653.data_ptr()), c_void_p(buf654.data_ptr()), c_void_p(buf655.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf657.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf659.data_ptr()), c_void_p(buf660.data_ptr()), c_void_p(buf661.data_ptr()), c_void_p(buf662.data_ptr()), c_void_p(buf663.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf665.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf667.data_ptr()), c_void_p(buf668.data_ptr()), c_void_p(buf669.data_ptr()), c_void_p(buf670.data_ptr()), c_void_p(buf671.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf673.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf675.data_ptr()), c_void_p(buf676.data_ptr()), c_void_p(buf677.data_ptr()), c_void_p(buf678.data_ptr()), c_void_p(buf679.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf681.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf683.data_ptr()), c_void_p(buf684.data_ptr()), c_void_p(buf685.data_ptr()), c_void_p(buf686.data_ptr()), c_void_p(buf687.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf689.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf691.data_ptr()), c_void_p(buf692.data_ptr()), c_void_p(buf693.data_ptr()), c_void_p(buf694.data_ptr()), c_void_p(buf695.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf697.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf636.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(buf637.data_ptr()), c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(buf700.data_ptr()), c_void_p(buf701.data_ptr()), c_void_p(buf702.data_ptr()), c_void_p(buf703.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf705.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf707.data_ptr()), c_void_p(buf708.data_ptr()), c_void_p(buf709.data_ptr()), c_void_p(buf710.data_ptr()), c_void_p(buf711.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf713.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf715.data_ptr()), c_void_p(buf716.data_ptr()), c_void_p(buf717.data_ptr()), c_void_p(buf718.data_ptr()), c_void_p(buf719.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf721.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf723.data_ptr()), c_void_p(buf724.data_ptr()), c_void_p(buf725.data_ptr()), c_void_p(buf726.data_ptr()), c_void_p(buf727.data_ptr()))
    del buf104
    del buf126
    del buf14
    del buf149
    del buf171
    del buf194
    del buf216
    del buf239
    del buf261
    del buf284
    del buf306
    del buf329
    del buf351
    del buf36
    del buf374
    del buf396
    del buf419
    del buf441
    del buf464
    del buf486
    del buf509
    del buf531
    del buf554
    del buf576
    del buf59
    del buf637
    del buf638
    del buf640
    del buf81
    del primals_286
    del primals_289
    del primals_292
    del primals_295
    del primals_298
    del primals_301
    del primals_304
    del primals_307
    del primals_310
    del primals_313
    del primals_316
    del primals_319
    del primals_322
    del primals_325
    del primals_328
    del primals_331
    del primals_334
    del primals_337
    del primals_340
    del primals_343
    del primals_346
    del primals_349
    del primals_352
    del primals_355
    del primals_358
    del primals_361
    del primals_364
    del primals_367
    return (buf699, reinterpret_tensor(buf636, (1, 128, 50400), (6451200, 50400, 1), 0), buf9, reinterpret_tensor(buf8, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf31, reinterpret_tensor(buf30, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf54, reinterpret_tensor(buf53, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf76, reinterpret_tensor(buf75, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf99, reinterpret_tensor(buf98, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf121, reinterpret_tensor(buf120, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf144, reinterpret_tensor(buf143, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf166, reinterpret_tensor(buf165, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf189, reinterpret_tensor(buf188, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf211, reinterpret_tensor(buf210, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf234, reinterpret_tensor(buf233, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf256, reinterpret_tensor(buf255, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf279, reinterpret_tensor(buf278, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf301, reinterpret_tensor(buf300, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf324, reinterpret_tensor(buf323, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf346, reinterpret_tensor(buf345, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf369, reinterpret_tensor(buf368, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf391, reinterpret_tensor(buf390, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf414, reinterpret_tensor(buf413, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf436, reinterpret_tensor(buf435, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf459, reinterpret_tensor(buf458, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf481, reinterpret_tensor(buf480, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf504, reinterpret_tensor(buf503, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf526, reinterpret_tensor(buf525, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf549, reinterpret_tensor(buf548, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf571, reinterpret_tensor(buf570, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf594, reinterpret_tensor(buf593, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), buf616, reinterpret_tensor(buf615, (1, 16, 128, 256), (524288, 256, 4096, 1), 0), primals_2, primals_12, primals_22, primals_32, primals_42, primals_52, primals_62, primals_72, primals_82, primals_92, primals_102, primals_112, primals_122, primals_132, primals_142, primals_152, primals_162, primals_172, primals_182, primals_192, primals_202, primals_212, primals_222, primals_232, primals_242, primals_252, primals_262, primals_272, primals_282, primals_288, primals_291, primals_294, primals_297, primals_300, primals_303, primals_306, primals_309, primals_312, primals_315, primals_318, primals_321, primals_324, primals_327, primals_330, primals_333, primals_336, primals_339, primals_342, primals_345, primals_348, primals_351, primals_354, primals_357, primals_360, primals_363, primals_366, primals_369, primals_371, primals_370, buf0, buf1, buf4, buf5, reinterpret_tensor(buf700, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf700, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_287, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf17, buf19, buf20, buf21, buf26, buf27, reinterpret_tensor(buf701, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf701, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_290, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf39, buf41, buf42, buf43, buf49, buf50, reinterpret_tensor(buf702, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf702, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_293, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf62, buf64, buf65, buf66, buf71, buf72, reinterpret_tensor(buf703, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf703, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_296, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf84, buf86, buf87, buf88, buf94, buf95, reinterpret_tensor(buf704, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf704, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_299, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf107, buf109, buf110, buf111, buf116, buf117, reinterpret_tensor(buf705, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf705, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_302, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf129, buf131, buf132, buf133, buf139, buf140, reinterpret_tensor(buf706, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf706, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_305, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf152, buf154, buf155, buf156, buf161, buf162, reinterpret_tensor(buf707, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf707, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_308, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf174, buf176, buf177, buf178, buf184, buf185, reinterpret_tensor(buf708, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf708, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_311, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf197, buf199, buf200, buf201, buf206, buf207, reinterpret_tensor(buf709, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf709, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_314, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf219, buf221, buf222, buf223, buf229, buf230, reinterpret_tensor(buf710, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf710, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_317, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf242, buf244, buf245, buf246, buf251, buf252, reinterpret_tensor(buf711, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf711, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_320, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf264, buf266, buf267, buf268, buf274, buf275, reinterpret_tensor(buf712, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf712, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_323, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf287, buf289, buf290, buf291, buf296, buf297, reinterpret_tensor(buf713, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf713, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_326, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf309, buf311, buf312, buf313, buf319, buf320, reinterpret_tensor(buf714, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf714, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_329, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf332, buf334, buf335, buf336, buf341, buf342, reinterpret_tensor(buf715, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf715, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_332, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf354, buf356, buf357, buf358, buf364, buf365, reinterpret_tensor(buf716, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf716, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_335, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf377, buf379, buf380, buf381, buf386, buf387, reinterpret_tensor(buf717, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf717, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_338, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf399, buf401, buf402, buf403, buf409, buf410, reinterpret_tensor(buf718, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf718, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_341, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf422, buf424, buf425, buf426, buf431, buf432, reinterpret_tensor(buf719, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf719, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_344, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf444, buf446, buf447, buf448, buf454, buf455, reinterpret_tensor(buf720, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf720, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_347, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf467, buf469, buf470, buf471, buf476, buf477, reinterpret_tensor(buf721, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf721, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_350, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf489, buf491, buf492, buf493, buf499, buf500, reinterpret_tensor(buf722, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf722, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_353, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf512, buf514, buf515, buf516, buf521, buf522, reinterpret_tensor(buf723, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf723, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_356, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf534, buf536, buf537, buf538, buf544, buf545, reinterpret_tensor(buf724, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf724, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_359, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf557, buf559, buf560, buf561, buf566, buf567, reinterpret_tensor(buf725, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf725, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_362, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf579, buf581, buf582, buf583, buf589, buf590, reinterpret_tensor(buf726, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf726, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_365, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf602, buf604, buf605, buf606, buf611, buf612, reinterpret_tensor(buf727, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 0), reinterpret_tensor(buf727, (1, 128, 1, 32, 1), (0, 64, 0, 1, 0), 32), reinterpret_tensor(primals_368, (1, 1, 128, 128), (4194304, 4194304, 2048, 1), 0), buf624, buf626, buf627, buf628, buf634, buf635, buf639, buf641, reinterpret_tensor(primals_284, (50400, 4096), (4096, 1), 0), buf643, reinterpret_tensor(primals_280, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_278, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_277, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf622, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf615, (16, 256, 128), (256, 1, 4096), 0), buf644, reinterpret_tensor(buf617, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf616, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_276, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_275, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_274, (4096, 4096), (4096, 1), 0), buf645, reinterpret_tensor(primals_270, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_268, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_267, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf600, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf593, (16, 256, 128), (256, 1, 4096), 0), buf646, reinterpret_tensor(buf595, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf594, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_266, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_265, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_264, (4096, 4096), (4096, 1), 0), buf647, reinterpret_tensor(primals_260, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_258, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_257, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf577, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf570, (16, 256, 128), (256, 1, 4096), 0), buf648, reinterpret_tensor(buf572, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf571, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_256, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_255, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_254, (4096, 4096), (4096, 1), 0), buf649, reinterpret_tensor(primals_250, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_248, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_247, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf555, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf548, (16, 256, 128), (256, 1, 4096), 0), buf650, reinterpret_tensor(buf550, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf549, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_246, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_245, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_244, (4096, 4096), (4096, 1), 0), buf651, reinterpret_tensor(primals_240, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_238, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_237, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf532, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf525, (16, 256, 128), (256, 1, 4096), 0), buf652, reinterpret_tensor(buf527, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf526, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_236, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_235, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_234, (4096, 4096), (4096, 1), 0), buf653, reinterpret_tensor(primals_230, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_228, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_227, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf510, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf503, (16, 256, 128), (256, 1, 4096), 0), buf654, reinterpret_tensor(buf505, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf504, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_226, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_225, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_224, (4096, 4096), (4096, 1), 0), buf655, reinterpret_tensor(primals_220, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_218, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_217, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf487, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf480, (16, 256, 128), (256, 1, 4096), 0), buf656, reinterpret_tensor(buf482, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf481, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_216, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_215, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_214, (4096, 4096), (4096, 1), 0), buf657, reinterpret_tensor(primals_210, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_208, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_207, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf465, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf458, (16, 256, 128), (256, 1, 4096), 0), buf658, reinterpret_tensor(buf460, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf459, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_206, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_205, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_204, (4096, 4096), (4096, 1), 0), buf659, reinterpret_tensor(primals_200, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_198, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_197, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf442, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf435, (16, 256, 128), (256, 1, 4096), 0), buf660, reinterpret_tensor(buf437, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf436, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_196, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_195, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_194, (4096, 4096), (4096, 1), 0), buf661, reinterpret_tensor(primals_190, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_188, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_187, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf420, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf413, (16, 256, 128), (256, 1, 4096), 0), buf662, reinterpret_tensor(buf415, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf414, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_186, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_185, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_184, (4096, 4096), (4096, 1), 0), buf663, reinterpret_tensor(primals_180, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_178, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_177, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf397, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf390, (16, 256, 128), (256, 1, 4096), 0), buf664, reinterpret_tensor(buf392, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf391, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_176, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_175, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_174, (4096, 4096), (4096, 1), 0), buf665, reinterpret_tensor(primals_170, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_168, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_167, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf375, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf368, (16, 256, 128), (256, 1, 4096), 0), buf666, reinterpret_tensor(buf370, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf369, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_166, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_165, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_164, (4096, 4096), (4096, 1), 0), buf667, reinterpret_tensor(primals_160, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_158, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_157, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf352, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf345, (16, 256, 128), (256, 1, 4096), 0), buf668, reinterpret_tensor(buf347, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf346, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_156, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_155, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_154, (4096, 4096), (4096, 1), 0), buf669, reinterpret_tensor(primals_150, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_148, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_147, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf330, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf323, (16, 256, 128), (256, 1, 4096), 0), buf670, reinterpret_tensor(buf325, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf324, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_146, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_145, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_144, (4096, 4096), (4096, 1), 0), buf671, reinterpret_tensor(primals_140, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_138, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_137, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf307, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf300, (16, 256, 128), (256, 1, 4096), 0), buf672, reinterpret_tensor(buf302, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf301, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_136, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_135, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_134, (4096, 4096), (4096, 1), 0), buf673, reinterpret_tensor(primals_130, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_128, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_127, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf285, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf278, (16, 256, 128), (256, 1, 4096), 0), buf674, reinterpret_tensor(buf280, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf279, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_126, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_125, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_124, (4096, 4096), (4096, 1), 0), buf675, reinterpret_tensor(primals_120, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_118, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_117, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf262, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf255, (16, 256, 128), (256, 1, 4096), 0), buf676, reinterpret_tensor(buf257, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf256, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_116, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_115, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_114, (4096, 4096), (4096, 1), 0), buf677, reinterpret_tensor(primals_110, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_108, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_107, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf240, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf233, (16, 256, 128), (256, 1, 4096), 0), buf678, reinterpret_tensor(buf235, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf234, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_106, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_105, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_104, (4096, 4096), (4096, 1), 0), buf679, reinterpret_tensor(primals_100, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_98, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_97, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf217, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf210, (16, 256, 128), (256, 1, 4096), 0), buf680, reinterpret_tensor(buf212, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf211, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_96, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_95, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_94, (4096, 4096), (4096, 1), 0), buf681, reinterpret_tensor(primals_90, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_88, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_87, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf195, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf188, (16, 256, 128), (256, 1, 4096), 0), buf682, reinterpret_tensor(buf190, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf189, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_86, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_85, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_84, (4096, 4096), (4096, 1), 0), buf683, reinterpret_tensor(primals_80, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_78, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_77, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf172, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf165, (16, 256, 128), (256, 1, 4096), 0), buf684, reinterpret_tensor(buf167, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf166, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_76, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_75, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_74, (4096, 4096), (4096, 1), 0), buf685, reinterpret_tensor(primals_70, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_68, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_67, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf150, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf143, (16, 256, 128), (256, 1, 4096), 0), buf686, reinterpret_tensor(buf145, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf144, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_66, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_65, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_64, (4096, 4096), (4096, 1), 0), buf687, reinterpret_tensor(primals_60, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_58, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_57, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf127, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf120, (16, 256, 128), (256, 1, 4096), 0), buf688, reinterpret_tensor(buf122, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf121, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_56, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_55, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_54, (4096, 4096), (4096, 1), 0), buf689, reinterpret_tensor(primals_50, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_48, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_47, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf105, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf98, (16, 256, 128), (256, 1, 4096), 0), buf690, reinterpret_tensor(buf100, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf99, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_46, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_45, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_44, (4096, 4096), (4096, 1), 0), buf691, reinterpret_tensor(primals_40, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_38, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_37, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf82, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf75, (16, 256, 128), (256, 1, 4096), 0), buf692, reinterpret_tensor(buf77, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf76, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_36, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_35, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_34, (4096, 4096), (4096, 1), 0), buf693, reinterpret_tensor(primals_30, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_28, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_27, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf60, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf53, (16, 256, 128), (256, 1, 4096), 0), buf694, reinterpret_tensor(buf55, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf54, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_26, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_25, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_24, (4096, 4096), (4096, 1), 0), buf695, reinterpret_tensor(primals_20, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_18, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_17, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf37, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf30, (16, 256, 128), (256, 1, 4096), 0), buf696, reinterpret_tensor(buf32, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf31, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_16, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_15, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_14, (4096, 4096), (4096, 1), 0), buf697, reinterpret_tensor(primals_10, (4096, 16384), (16384, 1), 0), reinterpret_tensor(primals_8, (16384, 4096), (4096, 1), 0), reinterpret_tensor(primals_7, (4096, 4096), (4096, 1), 0), reinterpret_tensor(buf15, (16, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf8, (16, 256, 128), (256, 1, 4096), 0), buf698, reinterpret_tensor(buf10, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf9, (16, 128, 256), (256, 4096, 1), 0), reinterpret_tensor(primals_6, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_5, (4096, 4096), (4096, 1), 0), reinterpret_tensor(primals_4, (4096, 4096), (4096, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((50400, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((4096, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((16384, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((16384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((4096, 16384), (16384, 1), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((50400, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((50400, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_288 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_291 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_294 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_297 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_300 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_303 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_306 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_309 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_312 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_315 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_318 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_321 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_324 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_327 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_330 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_333 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_336 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_339 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_342 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_345 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_348 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_351 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_354 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_357 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_360 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_363 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_366 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((2048, 64), (64, 1), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cpu', dtype=torch.bool)
    primals_369 = rand_strided((), (), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_371 = rand_strided((1, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('GPTJForCausalLM', benchmark_compiled_module)
