
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
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (196L*x1) + (588L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (588L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_1 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(962L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2)];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(962);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(961L))) + (246016L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((961L*x2) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(961L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>(x2)];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(961L))) + (246016L*x0))];
                                auto tmp22 = in_ptr2[static_cast<long>((961L*x2) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(961L)))];
                                auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp25 = tmp4 ? tmp19 : tmp24;
                            tmp_acc0 = welford_combine(tmp_acc0, tmp16);
                            tmp_acc1 = welford_combine(tmp_acc1, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (962L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (962L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(962L); x2+=static_cast<long>(1L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x2 + (962L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x2 + (962L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(256.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x1 + (256L*x2) + (246272L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_2 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(962L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                        {
                            auto tmp17 = in_ptr3[static_cast<long>(x2 + (256L*x1) + (246272L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>(x2)];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(962);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(961L))) + (246016L*x0))];
                                auto tmp13 = in_ptr2[static_cast<long>((961L*x2) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(961L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>(x2)];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp4 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr1[static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(961L))) + (246016L*x0))];
                                auto tmp24 = in_ptr2[static_cast<long>((961L*x2) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(961L)))];
                                auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                                return tmp25;
                            }
                            ;
                            auto tmp26 = tmp8 ? tmp22() : static_cast<decltype(tmp22())>(0.0);
                            auto tmp27 = tmp4 ? tmp21 : tmp26;
                            auto tmp28 = decltype(tmp27)(tmp27 + tmp17);
                            tmp_acc0 = welford_combine(tmp_acc0, tmp18);
                            tmp_acc1 = welford_combine(tmp_acc1, tmp28);
                        }
                        out_ptr0[static_cast<long>(x1 + (962L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (962L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(962L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = in_ptr3[static_cast<long>(x2 + (256L*x1) + (246272L*x0))];
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (962L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (962L*x0))];
                        auto tmp28 = in_ptr4[static_cast<long>(x2)];
                        auto tmp30 = in_ptr5[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2)];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(962);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(961L))) + (246016L*x0))];
                            auto tmp13 = in_ptr2[static_cast<long>((961L*x2) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(961L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                        auto tmp22 = static_cast<float>(256.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                        out_ptr2[static_cast<long>(x2 + (256L*x1) + (246272L*x0))] = tmp31;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_4 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(962L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (246272L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (246272L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(962);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (256L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(961L))) + (246016L*x0)), to_float_mask(tmp8));
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr2[static_cast<long>((961L*x2) + (961L*x2_inner) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(961L)))]; return masked_load(tmpbuf, to_float_mask(tmp8)); })();
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (256L*x1) + (246272L*x0)));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_5 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_8 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(7880704L); x0+=static_cast<long>(8L))
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


cpp_fused_add_10 = async_compile.cpp('''
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
    }
}
''')


cpp_fused_cat_native_layer_norm_11 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
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
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (512L*x0)), to_float_mask(tmp4));
                                auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                                auto tmp21 = tmp19 + tmp20;
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr2 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(256L))) + (131072L*x0)), to_float_mask(tmp10));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp10));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (257L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (257L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (257L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (257L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(512.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_12 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
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
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (512L*x0)), to_float_mask(tmp4));
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                                auto tmp23 = tmp21 + tmp22;
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp25 = [&]
                            {
                                auto tmp26 = masked_load(in_ptr2 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(256L))) + (131072L*x0)), to_float_mask(tmp10));
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp10));
                            auto tmp28 = decltype(tmp24)::blendv(tmp27, tmp24, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (257L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (257L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (257L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (257L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(512.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_14 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(257L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
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
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (131584L*x0)));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_15 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_17 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
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
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_21 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_23 = async_compile.cpp('''
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
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
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_26 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_27 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4210688L); x0+=static_cast<long>(8L))
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


cpp_fused_add_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1052672L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_native_layer_norm_30 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
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
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (1024L*x0)), to_float_mask(tmp4));
                                auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                                auto tmp21 = tmp19 + tmp20;
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr2 + static_cast<long>(x2 + (1024L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(64L))) + (65536L*x0)), to_float_mask(tmp10));
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp10));
                            auto tmp26 = decltype(tmp22)::blendv(tmp25, tmp22, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (65L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (65L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (65L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (65L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(1024.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
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
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (1024L*x0)), to_float_mask(tmp4));
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp4));
                                auto tmp23 = tmp21 + tmp22;
                                return tmp23;
                            }
                            ;
                            auto tmp24 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp25 = [&]
                            {
                                auto tmp26 = masked_load(in_ptr2 + static_cast<long>(x2 + (1024L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(64L))) + (65536L*x0)), to_float_mask(tmp10));
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp25())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp25(), to_float_mask(tmp10));
                            auto tmp28 = decltype(tmp24)::blendv(tmp27, tmp24, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (65L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (65L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (65L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (65L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(1024.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(65L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
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
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (66560L*x0)));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_37 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_39 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2129920L); x0+=static_cast<long>(8L))
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


cpp_fused_native_layer_norm_42 = async_compile.cpp('''
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (66560L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (66560L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (66560L*x0)));
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
                tmp19.store(out_ptr2 + static_cast<long>(x1 + (1024L*x0)));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 256, 31, 31), (246016, 961, 31, 1))
    assert_size_stride(arg1_1, (1, 1, 256), (256, 256, 1))
    assert_size_stride(arg2_1, (256, 3, 14, 14), (588, 196, 14, 1))
    assert_size_stride(arg3_1, (256, ), (1, ))
    assert_size_stride(arg4_1, (256, ), (1, ))
    assert_size_stride(arg5_1, (256, ), (1, ))
    assert_size_stride(arg6_1, (768, 256), (256, 1))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (256, 256), (256, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (1024, 256), (256, 1))
    assert_size_stride(arg13_1, (1024, ), (1, ))
    assert_size_stride(arg14_1, (256, 1024), (1024, 1))
    assert_size_stride(arg15_1, (256, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (768, 256), (256, 1))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (256, 256), (256, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (1024, 256), (256, 1))
    assert_size_stride(arg25_1, (1024, ), (1, ))
    assert_size_stride(arg26_1, (256, 1024), (1024, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, ), (1, ))
    assert_size_stride(arg30_1, (768, 256), (256, 1))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (256, 256), (256, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (1024, 256), (256, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (256, 1024), (1024, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 256), (256, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, ), (1, ))
    assert_size_stride(arg46_1, (1536, 512), (512, 1))
    assert_size_stride(arg47_1, (1536, ), (1, ))
    assert_size_stride(arg48_1, (512, 512), (512, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, ), (1, ))
    assert_size_stride(arg52_1, (2048, 512), (512, 1))
    assert_size_stride(arg53_1, (2048, ), (1, ))
    assert_size_stride(arg54_1, (512, 2048), (2048, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (1536, 512), (512, 1))
    assert_size_stride(arg59_1, (1536, ), (1, ))
    assert_size_stride(arg60_1, (512, 512), (512, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (2048, 512), (512, 1))
    assert_size_stride(arg65_1, (2048, ), (1, ))
    assert_size_stride(arg66_1, (512, 2048), (2048, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (1536, 512), (512, 1))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (512, 512), (512, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (2048, 512), (512, 1))
    assert_size_stride(arg77_1, (2048, ), (1, ))
    assert_size_stride(arg78_1, (512, 2048), (2048, 1))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (1536, 512), (512, 1))
    assert_size_stride(arg83_1, (1536, ), (1, ))
    assert_size_stride(arg84_1, (512, 512), (512, 1))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (2048, 512), (512, 1))
    assert_size_stride(arg89_1, (2048, ), (1, ))
    assert_size_stride(arg90_1, (512, 2048), (2048, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (1536, 512), (512, 1))
    assert_size_stride(arg95_1, (1536, ), (1, ))
    assert_size_stride(arg96_1, (512, 512), (512, 1))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (2048, 512), (512, 1))
    assert_size_stride(arg101_1, (2048, ), (1, ))
    assert_size_stride(arg102_1, (512, 2048), (2048, 1))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (1536, 512), (512, 1))
    assert_size_stride(arg107_1, (1536, ), (1, ))
    assert_size_stride(arg108_1, (512, 512), (512, 1))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (512, ), (1, ))
    assert_size_stride(arg111_1, (512, ), (1, ))
    assert_size_stride(arg112_1, (2048, 512), (512, 1))
    assert_size_stride(arg113_1, (2048, ), (1, ))
    assert_size_stride(arg114_1, (512, 2048), (2048, 1))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg117_1, (1024, ), (1, ))
    assert_size_stride(arg118_1, (1024, 512), (512, 1))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (1024, ), (1, ))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg123_1, (3072, ), (1, ))
    assert_size_stride(arg124_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg125_1, (1024, ), (1, ))
    assert_size_stride(arg126_1, (1024, ), (1, ))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg129_1, (4096, ), (1, ))
    assert_size_stride(arg130_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg131_1, (1024, ), (1, ))
    assert_size_stride(arg132_1, (1024, ), (1, ))
    assert_size_stride(arg133_1, (1024, ), (1, ))
    assert_size_stride(arg134_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg135_1, (3072, ), (1, ))
    assert_size_stride(arg136_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (1024, ), (1, ))
    assert_size_stride(arg139_1, (1024, ), (1, ))
    assert_size_stride(arg140_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg141_1, (4096, ), (1, ))
    assert_size_stride(arg142_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg143_1, (1024, ), (1, ))
    assert_size_stride(arg144_1, (1024, ), (1, ))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg147_1, (3072, ), (1, ))
    assert_size_stride(arg148_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg149_1, (1024, ), (1, ))
    assert_size_stride(arg150_1, (1024, ), (1, ))
    assert_size_stride(arg151_1, (1024, ), (1, ))
    assert_size_stride(arg152_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg153_1, (4096, ), (1, ))
    assert_size_stride(arg154_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (1024, ), (1, ))
    assert_size_stride(arg157_1, (1024, ), (1, ))
    assert_size_stride(arg158_1, (3072, 1024), (1024, 1))
    assert_size_stride(arg159_1, (3072, ), (1, ))
    assert_size_stride(arg160_1, (1024, 1024), (1024, 1))
    assert_size_stride(arg161_1, (1024, ), (1, ))
    assert_size_stride(arg162_1, (1024, ), (1, ))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (4096, 1024), (1024, 1))
    assert_size_stride(arg165_1, (4096, ), (1, ))
    assert_size_stride(arg166_1, (1024, 4096), (4096, 1))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1024, ), (1, ))
    assert_size_stride(arg169_1, (1024, ), (1, ))
    assert_size_stride(arg170_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg171_1, (1000, ), (1, ))
    assert_size_stride(arg172_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((256, 3, 14, 14), (588, 1, 42, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg172_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg172_1
    del arg2_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg3_1, stride=(7, 7), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 256, 31, 31), (246016, 1, 7936, 256))
    del arg3_1
    del buf0
    del buf1
    buf3 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 962, 1), (962, 1, 7696), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_1(c_void_p(arg1_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg4_1
    del arg5_1
    buf7 = empty((7696, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_0_blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg7_1, reinterpret_tensor(buf6, (7696, 256), (256, 1), 0), reinterpret_tensor(arg6_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf7)
    del arg6_1
    del arg7_1
    # Source Nodes: [x_6], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf8 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf7, (8, 4, 962, 64), (738816, 64, 768, 1), 512))
    buf9 = buf8[0]
    del buf8
    buf16 = reinterpret_tensor(buf6, (7696, 256), (256, 1), 0); del buf6  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg9_1, reinterpret_tensor(buf9, (7696, 256), (256, 1), 0), reinterpret_tensor(arg8_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf16)
    del arg8_1
    del arg9_1
    buf17 = buf4; del buf4  # reuse
    buf18 = buf3; del buf3  # reuse
    buf20 = reinterpret_tensor(buf9, (8, 962, 256), (246272, 256, 1), 0); del buf9  # reuse
    cpp_fused_add_cat_native_layer_norm_2(c_void_p(arg1_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg10_1
    del arg11_1
    buf21 = empty((7696, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg13_1, reinterpret_tensor(buf20, (7696, 256), (256, 1), 0), reinterpret_tensor(arg12_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf21)
    del arg12_1
    del arg13_1
    buf22 = reinterpret_tensor(buf21, (8, 962, 1024), (985088, 1024, 1), 0); del buf21  # reuse
    cpp_fused_gelu_3(c_void_p(buf22.data_ptr()))
    buf23 = reinterpret_tensor(buf20, (7696, 256), (256, 1), 0); del buf20  # reuse
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg15_1, reinterpret_tensor(buf22, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg14_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf23)
    del arg14_1
    del arg15_1
    buf24 = reinterpret_tensor(buf23, (8, 962, 256), (246272, 256, 1), 0); del buf23  # reuse
    buf25 = buf18; del buf18  # reuse
    buf26 = buf17; del buf17  # reuse
    buf28 = empty((8, 962, 256), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_4(c_void_p(buf24.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg0_1
    del arg16_1
    del arg17_1
    del arg1_1
    del buf2
    buf29 = buf7; del buf7  # reuse
    # Source Nodes: [getattr_l__mod___transformers_0_blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg19_1, reinterpret_tensor(buf28, (7696, 256), (256, 1), 0), reinterpret_tensor(arg18_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf29)
    del arg18_1
    del arg19_1
    # Source Nodes: [x_18], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf30 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf29, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf29, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf29, (8, 4, 962, 64), (738816, 64, 768, 1), 512))
    buf31 = buf30[0]
    del buf30
    buf38 = reinterpret_tensor(buf28, (7696, 256), (256, 1), 0); del buf28  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg21_1, reinterpret_tensor(buf31, (7696, 256), (256, 1), 0), reinterpret_tensor(arg20_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf38)
    del arg20_1
    del arg21_1
    buf39 = buf26; del buf26  # reuse
    buf40 = buf25; del buf25  # reuse
    buf42 = reinterpret_tensor(buf31, (8, 962, 256), (246272, 256, 1), 0); del buf31  # reuse
    cpp_fused_add_native_layer_norm_5(c_void_p(buf24.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg22_1
    del arg23_1
    buf43 = reinterpret_tensor(buf22, (7696, 1024), (1024, 1), 0); del buf22  # reuse
    # Source Nodes: [x_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf42, (7696, 256), (256, 1), 0), reinterpret_tensor(arg24_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf43)
    del arg24_1
    del arg25_1
    buf44 = reinterpret_tensor(buf43, (8, 962, 1024), (985088, 1024, 1), 0); del buf43  # reuse
    cpp_fused_gelu_6(c_void_p(buf44.data_ptr()))
    buf45 = reinterpret_tensor(buf42, (7696, 256), (256, 1), 0); del buf42  # reuse
    # Source Nodes: [x_27], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg27_1, reinterpret_tensor(buf44, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg26_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf45)
    del arg26_1
    del arg27_1
    buf46 = buf40; del buf40  # reuse
    buf47 = buf39; del buf39  # reuse
    buf49 = reinterpret_tensor(buf16, (8, 962, 256), (246272, 256, 1), 0); del buf16  # reuse
    cpp_fused_add_native_layer_norm_7(c_void_p(buf24.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf49.data_ptr()))
    del arg28_1
    del arg29_1
    buf50 = buf29; del buf29  # reuse
    # Source Nodes: [getattr_l__mod___transformers_0_blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg31_1, reinterpret_tensor(buf49, (7696, 256), (256, 1), 0), reinterpret_tensor(arg30_1, (256, 768), (1, 256), 0), alpha=1, beta=1, out=buf50)
    del arg30_1
    del arg31_1
    # Source Nodes: [x_30], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf51 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf50, (8, 4, 962, 64), (738816, 64, 768, 1), 0), reinterpret_tensor(buf50, (8, 4, 962, 64), (738816, 64, 768, 1), 256), reinterpret_tensor(buf50, (8, 4, 962, 64), (738816, 64, 768, 1), 512))
    del buf50
    buf52 = buf51[0]
    del buf51
    buf59 = reinterpret_tensor(buf49, (7696, 256), (256, 1), 0); del buf49  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg33_1, reinterpret_tensor(buf52, (7696, 256), (256, 1), 0), reinterpret_tensor(arg32_1, (256, 256), (1, 256), 0), alpha=1, beta=1, out=buf59)
    del arg32_1
    del arg33_1
    buf60 = buf47; del buf47  # reuse
    buf61 = buf46; del buf46  # reuse
    buf63 = reinterpret_tensor(buf52, (8, 962, 256), (246272, 256, 1), 0); del buf52  # reuse
    cpp_fused_add_native_layer_norm_8(c_void_p(buf24.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf63.data_ptr()))
    del arg34_1
    del arg35_1
    del buf60
    del buf61
    buf64 = reinterpret_tensor(buf44, (7696, 1024), (1024, 1), 0); del buf44  # reuse
    # Source Nodes: [x_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg37_1, reinterpret_tensor(buf63, (7696, 256), (256, 1), 0), reinterpret_tensor(arg36_1, (256, 1024), (1, 256), 0), alpha=1, beta=1, out=buf64)
    del arg36_1
    del arg37_1
    buf65 = reinterpret_tensor(buf64, (8, 962, 1024), (985088, 1024, 1), 0); del buf64  # reuse
    cpp_fused_gelu_9(c_void_p(buf65.data_ptr()))
    buf66 = reinterpret_tensor(buf63, (7696, 256), (256, 1), 0); del buf63  # reuse
    # Source Nodes: [x_39], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg39_1, reinterpret_tensor(buf65, (7696, 1024), (1024, 1), 0), reinterpret_tensor(arg38_1, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf66)
    del arg38_1
    del arg39_1
    del buf65
    buf67 = reinterpret_tensor(buf66, (8, 962, 256), (246272, 256, 1), 0); del buf66  # reuse
    cpp_fused_add_10(c_void_p(buf67.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf59.data_ptr()))
    del buf24
    del buf38
    del buf45
    del buf59
    buf68 = empty((8, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [cls_tokens_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf67, (8, 256), (246272, 1), 0), reinterpret_tensor(arg42_1, (256, 512), (1, 256), 0), out=buf68)
    del arg42_1
    # Source Nodes: [x_47], Original ATen: [aten.convolution]
    buf69 = extern_kernels.convolution(reinterpret_tensor(buf67, (8, 256, 31, 31), (246272, 1, 7936, 256), 256), arg40_1, arg41_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256)
    assert_size_stride(buf69, (8, 512, 16, 16), (131072, 1, 8192, 512))
    del arg40_1
    del arg41_1
    del buf67
    buf70 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf71 = empty_strided((8, 257, 1), (257, 1, 2056), device='cpu', dtype=torch.float32)
    buf73 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_11(c_void_p(buf68.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()))
    del arg44_1
    del arg45_1
    buf74 = empty((2056, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg47_1, reinterpret_tensor(buf73, (2056, 512), (512, 1), 0), reinterpret_tensor(arg46_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf74)
    del arg46_1
    del arg47_1
    # Source Nodes: [x_51], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf75 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf74, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf76 = buf75[0]
    del buf75
    buf83 = reinterpret_tensor(buf73, (2056, 512), (512, 1), 0); del buf73  # reuse
    # Source Nodes: [x_53], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf76, (2056, 512), (512, 1), 0), reinterpret_tensor(arg48_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf83)
    del arg48_1
    del arg49_1
    buf84 = buf71; del buf71  # reuse
    buf85 = buf70; del buf70  # reuse
    buf87 = reinterpret_tensor(buf76, (8, 257, 512), (131584, 512, 1), 0); del buf76  # reuse
    cpp_fused_add_cat_native_layer_norm_12(c_void_p(buf68.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf87.data_ptr()))
    del arg50_1
    del arg51_1
    buf88 = empty((2056, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_56], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf87, (2056, 512), (512, 1), 0), reinterpret_tensor(arg52_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf88)
    del arg52_1
    del arg53_1
    buf89 = reinterpret_tensor(buf88, (8, 257, 2048), (526336, 2048, 1), 0); del buf88  # reuse
    cpp_fused_gelu_13(c_void_p(buf89.data_ptr()))
    buf90 = reinterpret_tensor(buf87, (2056, 512), (512, 1), 0); del buf87  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf89, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg54_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf90)
    del arg54_1
    del arg55_1
    buf91 = reinterpret_tensor(buf90, (8, 257, 512), (131584, 512, 1), 0); del buf90  # reuse
    buf92 = buf85; del buf85  # reuse
    buf93 = buf84; del buf84  # reuse
    buf95 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_14(c_void_p(buf91.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()))
    del arg43_1
    del arg56_1
    del arg57_1
    del buf68
    del buf69
    buf96 = buf74; del buf74  # reuse
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf95, (2056, 512), (512, 1), 0), reinterpret_tensor(arg58_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf96)
    del arg58_1
    del arg59_1
    # Source Nodes: [x_63], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf97 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf96, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf96, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf96, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf98 = buf97[0]
    del buf97
    buf105 = reinterpret_tensor(buf95, (2056, 512), (512, 1), 0); del buf95  # reuse
    # Source Nodes: [x_65], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg61_1, reinterpret_tensor(buf98, (2056, 512), (512, 1), 0), reinterpret_tensor(arg60_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf105)
    del arg60_1
    del arg61_1
    buf106 = buf93; del buf93  # reuse
    buf107 = buf92; del buf92  # reuse
    buf109 = reinterpret_tensor(buf98, (8, 257, 512), (131584, 512, 1), 0); del buf98  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf91.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg62_1
    del arg63_1
    buf110 = reinterpret_tensor(buf89, (2056, 2048), (2048, 1), 0); del buf89  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg65_1, reinterpret_tensor(buf109, (2056, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf110)
    del arg64_1
    del arg65_1
    buf111 = reinterpret_tensor(buf110, (8, 257, 2048), (526336, 2048, 1), 0); del buf110  # reuse
    cpp_fused_gelu_16(c_void_p(buf111.data_ptr()))
    buf112 = reinterpret_tensor(buf109, (2056, 512), (512, 1), 0); del buf109  # reuse
    # Source Nodes: [x_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg67_1, reinterpret_tensor(buf111, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg66_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf112)
    del arg66_1
    del arg67_1
    buf113 = buf107; del buf107  # reuse
    buf114 = buf106; del buf106  # reuse
    buf116 = reinterpret_tensor(buf83, (8, 257, 512), (131584, 512, 1), 0); del buf83  # reuse
    cpp_fused_add_native_layer_norm_17(c_void_p(buf91.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg68_1
    del arg69_1
    buf117 = buf96; del buf96  # reuse
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf116, (2056, 512), (512, 1), 0), reinterpret_tensor(arg70_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf117)
    del arg70_1
    del arg71_1
    # Source Nodes: [x_75], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf118 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf117, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf117, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf117, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf119 = buf118[0]
    del buf118
    buf126 = reinterpret_tensor(buf116, (2056, 512), (512, 1), 0); del buf116  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf119, (2056, 512), (512, 1), 0), reinterpret_tensor(arg72_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf126)
    del arg72_1
    del arg73_1
    buf127 = buf114; del buf114  # reuse
    buf128 = buf113; del buf113  # reuse
    buf130 = reinterpret_tensor(buf119, (8, 257, 512), (131584, 512, 1), 0); del buf119  # reuse
    cpp_fused_add_native_layer_norm_18(c_void_p(buf91.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg74_1
    del arg75_1
    buf131 = reinterpret_tensor(buf111, (2056, 2048), (2048, 1), 0); del buf111  # reuse
    # Source Nodes: [x_80], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg77_1, reinterpret_tensor(buf130, (2056, 512), (512, 1), 0), reinterpret_tensor(arg76_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf131)
    del arg76_1
    del arg77_1
    buf132 = reinterpret_tensor(buf131, (8, 257, 2048), (526336, 2048, 1), 0); del buf131  # reuse
    cpp_fused_gelu_19(c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf130, (2056, 512), (512, 1), 0); del buf130  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf132, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg78_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf133)
    del arg78_1
    del arg79_1
    buf134 = reinterpret_tensor(buf133, (8, 257, 512), (131584, 512, 1), 0); del buf133  # reuse
    buf135 = buf128; del buf128  # reuse
    buf136 = buf127; del buf127  # reuse
    buf138 = empty((8, 257, 512), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_20(c_void_p(buf134.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()))
    del arg80_1
    del arg81_1
    del buf105
    del buf112
    buf139 = buf117; del buf117  # reuse
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, reinterpret_tensor(buf138, (2056, 512), (512, 1), 0), reinterpret_tensor(arg82_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf139)
    del arg82_1
    del arg83_1
    # Source Nodes: [x_87], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf140 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf139, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf139, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf139, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf141 = buf140[0]
    del buf140
    buf148 = reinterpret_tensor(buf138, (2056, 512), (512, 1), 0); del buf138  # reuse
    # Source Nodes: [x_89], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf141, (2056, 512), (512, 1), 0), reinterpret_tensor(arg84_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf148)
    del arg84_1
    del arg85_1
    buf149 = buf136; del buf136  # reuse
    buf150 = buf135; del buf135  # reuse
    buf152 = reinterpret_tensor(buf141, (8, 257, 512), (131584, 512, 1), 0); del buf141  # reuse
    cpp_fused_add_native_layer_norm_21(c_void_p(buf134.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()))
    del arg86_1
    del arg87_1
    buf153 = reinterpret_tensor(buf132, (2056, 2048), (2048, 1), 0); del buf132  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf152, (2056, 512), (512, 1), 0), reinterpret_tensor(arg88_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf153)
    del arg88_1
    del arg89_1
    buf154 = reinterpret_tensor(buf153, (8, 257, 2048), (526336, 2048, 1), 0); del buf153  # reuse
    cpp_fused_gelu_22(c_void_p(buf154.data_ptr()))
    buf155 = reinterpret_tensor(buf152, (2056, 512), (512, 1), 0); del buf152  # reuse
    # Source Nodes: [x_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf154, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg90_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf155)
    del arg90_1
    del arg91_1
    buf156 = buf150; del buf150  # reuse
    buf157 = buf149; del buf149  # reuse
    buf159 = buf91; del buf91  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf134.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()))
    del arg92_1
    del arg93_1
    buf160 = buf139; del buf139  # reuse
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___4___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf159, (2056, 512), (512, 1), 0), reinterpret_tensor(arg94_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf160)
    del arg94_1
    del arg95_1
    # Source Nodes: [x_99], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf161 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf160, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf160, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf160, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    buf162 = buf161[0]
    del buf161
    buf169 = reinterpret_tensor(buf159, (2056, 512), (512, 1), 0); del buf159  # reuse
    # Source Nodes: [x_101], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf162, (2056, 512), (512, 1), 0), reinterpret_tensor(arg96_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf169)
    del arg96_1
    del arg97_1
    buf170 = buf157; del buf157  # reuse
    buf171 = buf156; del buf156  # reuse
    buf173 = reinterpret_tensor(buf162, (8, 257, 512), (131584, 512, 1), 0); del buf162  # reuse
    cpp_fused_add_native_layer_norm_24(c_void_p(buf134.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()))
    del arg98_1
    del arg99_1
    buf174 = reinterpret_tensor(buf154, (2056, 2048), (2048, 1), 0); del buf154  # reuse
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf173, (2056, 512), (512, 1), 0), reinterpret_tensor(arg100_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf174)
    del arg100_1
    del arg101_1
    buf175 = reinterpret_tensor(buf174, (8, 257, 2048), (526336, 2048, 1), 0); del buf174  # reuse
    cpp_fused_gelu_25(c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf173, (2056, 512), (512, 1), 0); del buf173  # reuse
    # Source Nodes: [x_108], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf175, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg102_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf176)
    del arg102_1
    del arg103_1
    buf177 = reinterpret_tensor(buf176, (8, 257, 512), (131584, 512, 1), 0); del buf176  # reuse
    buf178 = buf171; del buf171  # reuse
    buf179 = buf170; del buf170  # reuse
    buf181 = reinterpret_tensor(buf126, (8, 257, 512), (131584, 512, 1), 0); del buf126  # reuse
    cpp_fused_add_native_layer_norm_26(c_void_p(buf177.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg104_1
    del arg105_1
    del buf134
    del buf148
    del buf155
    del buf169
    buf182 = buf160; del buf160  # reuse
    # Source Nodes: [getattr_l__mod___transformers_1_blocks___5___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf181, (2056, 512), (512, 1), 0), reinterpret_tensor(arg106_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf182)
    del arg106_1
    del arg107_1
    # Source Nodes: [x_111], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf183 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf182, (8, 8, 257, 64), (394752, 64, 1536, 1), 0), reinterpret_tensor(buf182, (8, 8, 257, 64), (394752, 64, 1536, 1), 512), reinterpret_tensor(buf182, (8, 8, 257, 64), (394752, 64, 1536, 1), 1024))
    del buf182
    buf184 = buf183[0]
    del buf183
    buf191 = reinterpret_tensor(buf181, (2056, 512), (512, 1), 0); del buf181  # reuse
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf184, (2056, 512), (512, 1), 0), reinterpret_tensor(arg108_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf191)
    del arg108_1
    del arg109_1
    buf192 = buf179; del buf179  # reuse
    buf193 = buf178; del buf178  # reuse
    buf195 = reinterpret_tensor(buf184, (8, 257, 512), (131584, 512, 1), 0); del buf184  # reuse
    cpp_fused_add_native_layer_norm_27(c_void_p(buf177.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del arg110_1
    del arg111_1
    del buf192
    del buf193
    buf196 = reinterpret_tensor(buf175, (2056, 2048), (2048, 1), 0); del buf175  # reuse
    # Source Nodes: [x_116], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf195, (2056, 512), (512, 1), 0), reinterpret_tensor(arg112_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf196)
    del arg112_1
    del arg113_1
    buf197 = reinterpret_tensor(buf196, (8, 257, 2048), (526336, 2048, 1), 0); del buf196  # reuse
    cpp_fused_gelu_28(c_void_p(buf197.data_ptr()))
    buf198 = reinterpret_tensor(buf195, (2056, 512), (512, 1), 0); del buf195  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf197, (2056, 2048), (2048, 1), 0), reinterpret_tensor(arg114_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf198)
    del arg114_1
    del arg115_1
    del buf197
    buf199 = reinterpret_tensor(buf198, (8, 257, 512), (131584, 512, 1), 0); del buf198  # reuse
    cpp_fused_add_29(c_void_p(buf199.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf191.data_ptr()))
    del buf177
    del buf191
    buf200 = empty((8, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [cls_tokens_6], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (8, 512), (131584, 1), 0), reinterpret_tensor(arg118_1, (512, 1024), (1, 512), 0), out=buf200)
    del arg118_1
    # Source Nodes: [x_128], Original ATen: [aten.convolution]
    buf201 = extern_kernels.convolution(reinterpret_tensor(buf199, (8, 512, 16, 16), (131584, 1, 8192, 512), 512), arg116_1, arg117_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf201, (8, 1024, 8, 8), (65536, 1, 8192, 1024))
    del arg116_1
    del arg117_1
    del buf199
    buf202 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf203 = empty_strided((8, 65, 1), (65, 1, 520), device='cpu', dtype=torch.float32)
    buf205 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_30(c_void_p(buf200.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()))
    del arg120_1
    del arg121_1
    buf206 = empty((520, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf205, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg122_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf206)
    del arg122_1
    del arg123_1
    # Source Nodes: [x_132], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf207 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf206, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf206, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf206, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    buf208 = buf207[0]
    del buf207
    buf215 = reinterpret_tensor(buf205, (520, 1024), (1024, 1), 0); del buf205  # reuse
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg125_1, reinterpret_tensor(buf208, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg124_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf215)
    del arg124_1
    del arg125_1
    buf216 = buf203; del buf203  # reuse
    buf217 = buf202; del buf202  # reuse
    buf219 = reinterpret_tensor(buf208, (8, 65, 1024), (66560, 1024, 1), 0); del buf208  # reuse
    cpp_fused_add_cat_native_layer_norm_31(c_void_p(buf200.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    del arg126_1
    del arg127_1
    buf220 = empty((520, 4096), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf219, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg128_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf220)
    del arg128_1
    del arg129_1
    buf221 = reinterpret_tensor(buf220, (8, 65, 4096), (266240, 4096, 1), 0); del buf220  # reuse
    cpp_fused_gelu_32(c_void_p(buf221.data_ptr()))
    buf222 = reinterpret_tensor(buf219, (520, 1024), (1024, 1), 0); del buf219  # reuse
    # Source Nodes: [x_141], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf221, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg130_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf222)
    del arg130_1
    del arg131_1
    buf223 = reinterpret_tensor(buf222, (8, 65, 1024), (66560, 1024, 1), 0); del buf222  # reuse
    buf224 = buf217; del buf217  # reuse
    buf225 = buf216; del buf216  # reuse
    buf227 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_33(c_void_p(buf223.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg119_1
    del arg132_1
    del arg133_1
    del buf201
    buf228 = buf206; del buf206  # reuse
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf227, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg134_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf228)
    del arg134_1
    del arg135_1
    # Source Nodes: [x_144], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf229 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf228, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf228, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf228, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    buf230 = buf229[0]
    del buf229
    buf237 = reinterpret_tensor(buf227, (520, 1024), (1024, 1), 0); del buf227  # reuse
    # Source Nodes: [x_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf230, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg136_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf237)
    del arg136_1
    del arg137_1
    buf238 = buf225; del buf225  # reuse
    buf239 = buf224; del buf224  # reuse
    buf241 = reinterpret_tensor(buf230, (8, 65, 1024), (66560, 1024, 1), 0); del buf230  # reuse
    cpp_fused_add_native_layer_norm_34(c_void_p(buf223.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()))
    del arg138_1
    del arg139_1
    buf242 = reinterpret_tensor(buf221, (520, 4096), (4096, 1), 0); del buf221  # reuse
    # Source Nodes: [x_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf241, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg140_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf242)
    del arg140_1
    del arg141_1
    buf243 = reinterpret_tensor(buf242, (8, 65, 4096), (266240, 4096, 1), 0); del buf242  # reuse
    cpp_fused_gelu_35(c_void_p(buf243.data_ptr()))
    buf244 = reinterpret_tensor(buf241, (520, 1024), (1024, 1), 0); del buf241  # reuse
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf243, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg142_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf244)
    del arg142_1
    del arg143_1
    buf245 = buf239; del buf239  # reuse
    buf246 = buf238; del buf238  # reuse
    buf248 = reinterpret_tensor(buf215, (8, 65, 1024), (66560, 1024, 1), 0); del buf215  # reuse
    cpp_fused_add_native_layer_norm_36(c_void_p(buf223.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()))
    del arg144_1
    del arg145_1
    buf249 = buf228; del buf228  # reuse
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg147_1, reinterpret_tensor(buf248, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg146_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf249)
    del arg146_1
    del arg147_1
    # Source Nodes: [x_156], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf250 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf249, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf249, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf249, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    buf251 = buf250[0]
    del buf250
    buf258 = reinterpret_tensor(buf248, (520, 1024), (1024, 1), 0); del buf248  # reuse
    # Source Nodes: [x_158], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg149_1, reinterpret_tensor(buf251, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg148_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf258)
    del arg148_1
    del arg149_1
    buf259 = buf246; del buf246  # reuse
    buf260 = buf245; del buf245  # reuse
    buf262 = reinterpret_tensor(buf251, (8, 65, 1024), (66560, 1024, 1), 0); del buf251  # reuse
    cpp_fused_add_native_layer_norm_37(c_void_p(buf223.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()))
    del arg150_1
    del arg151_1
    buf263 = reinterpret_tensor(buf243, (520, 4096), (4096, 1), 0); del buf243  # reuse
    # Source Nodes: [x_161], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf262, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg152_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf263)
    del arg152_1
    del arg153_1
    buf264 = reinterpret_tensor(buf263, (8, 65, 4096), (266240, 4096, 1), 0); del buf263  # reuse
    cpp_fused_gelu_38(c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf262, (520, 1024), (1024, 1), 0); del buf262  # reuse
    # Source Nodes: [x_165], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf264, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg154_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf265)
    del arg154_1
    del arg155_1
    buf266 = reinterpret_tensor(buf265, (8, 65, 1024), (66560, 1024, 1), 0); del buf265  # reuse
    buf267 = buf260; del buf260  # reuse
    buf268 = buf259; del buf259  # reuse
    buf270 = empty((8, 65, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_39(c_void_p(buf266.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()))
    del arg156_1
    del arg157_1
    del buf223
    del buf237
    del buf244
    del buf258
    buf271 = buf249; del buf249  # reuse
    # Source Nodes: [getattr_l__mod___transformers_2_blocks___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf270, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg158_1, (1024, 3072), (1, 1024), 0), alpha=1, beta=1, out=buf271)
    del arg158_1
    del arg159_1
    # Source Nodes: [x_168], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf272 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf271, (8, 16, 65, 64), (199680, 64, 3072, 1), 0), reinterpret_tensor(buf271, (8, 16, 65, 64), (199680, 64, 3072, 1), 1024), reinterpret_tensor(buf271, (8, 16, 65, 64), (199680, 64, 3072, 1), 2048))
    del buf271
    buf273 = buf272[0]
    del buf272
    buf280 = reinterpret_tensor(buf270, (520, 1024), (1024, 1), 0); del buf270  # reuse
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf273, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg160_1, (1024, 1024), (1, 1024), 0), alpha=1, beta=1, out=buf280)
    del arg160_1
    del arg161_1
    buf281 = buf268; del buf268  # reuse
    buf282 = buf267; del buf267  # reuse
    buf284 = reinterpret_tensor(buf273, (8, 65, 1024), (66560, 1024, 1), 0); del buf273  # reuse
    cpp_fused_add_native_layer_norm_40(c_void_p(buf266.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    del arg162_1
    del arg163_1
    del buf281
    del buf282
    buf285 = reinterpret_tensor(buf264, (520, 4096), (4096, 1), 0); del buf264  # reuse
    # Source Nodes: [x_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf284, (520, 1024), (1024, 1), 0), reinterpret_tensor(arg164_1, (1024, 4096), (1, 1024), 0), alpha=1, beta=1, out=buf285)
    del arg164_1
    del arg165_1
    buf286 = reinterpret_tensor(buf285, (8, 65, 4096), (266240, 4096, 1), 0); del buf285  # reuse
    cpp_fused_gelu_41(c_void_p(buf286.data_ptr()))
    buf287 = reinterpret_tensor(buf284, (520, 1024), (1024, 1), 0); del buf284  # reuse
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf286, (520, 4096), (4096, 1), 0), reinterpret_tensor(arg166_1, (4096, 1024), (1, 4096), 0), alpha=1, beta=1, out=buf287)
    del arg166_1
    del arg167_1
    del buf286
    buf288 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf291 = reinterpret_tensor(buf200, (8, 1, 1024), (1024, 1024, 1), 0); del buf200  # reuse
    cpp_fused_native_layer_norm_42(c_void_p(buf266.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del arg168_1
    del arg169_1
    del buf266
    del buf280
    del buf287
    del buf288
    del buf289
    buf292 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf291, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg170_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf292)
    del arg170_1
    del arg171_1
    return (buf292, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 256, 31, 31), (246016, 961, 31, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 256), (256, 256, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((256, 3, 14, 14), (588, 196, 14, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((256, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1024, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((3072, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((1024, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((4096, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((4096, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1024, 4096), (4096, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((1000, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pit_b_224', benchmark_compiled_module)
