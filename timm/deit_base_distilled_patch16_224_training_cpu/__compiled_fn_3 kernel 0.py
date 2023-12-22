
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (256L*x1) + (768L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (768L*x0))] = tmp0;
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


cpp_fused_add_cat_native_layer_norm_view_1 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(198L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1)));
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
                            auto tmp9 = static_cast<int>(2);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = tmp8 & tmp10;
                            auto tmp12 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp11));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                            auto tmp15 = tmp0 >= tmp9;
                            auto tmp16 = static_cast<int>(198);
                            auto tmp17 = tmp0 < tmp16;
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr2 + static_cast<long>(x2 + (768L*(static_cast<long>(((-2L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp15));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                            auto tmp21 = to_float_mask(tmp11);
                            auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                            auto tmp23 = to_float_mask(tmp4);
                            auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                            auto tmp26 = tmp24 + tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp28;
                            }
                            ;
                            auto tmp29 = decltype(tmp27())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp27(), to_float_mask(tmp4));
                            auto tmp30 = [&]
                            {
                                auto tmp31 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp11));
                                return tmp31;
                            }
                            ;
                            auto tmp32 = decltype(tmp30())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp30(), to_float_mask(tmp11));
                            auto tmp33 = [&]
                            {
                                auto tmp34 = masked_load(in_ptr2 + static_cast<long>(x2 + (768L*(static_cast<long>(((-2L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp15));
                                return tmp34;
                            }
                            ;
                            auto tmp35 = decltype(tmp33())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp33(), to_float_mask(tmp15));
                            auto tmp36 = decltype(tmp32)::blendv(tmp35, tmp32, tmp21);
                            auto tmp37 = decltype(tmp29)::blendv(tmp36, tmp29, tmp23);
                            auto tmp38 = tmp37 + tmp25;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp26);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp38);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (198L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (198L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(198L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp27 = out_ptr0[static_cast<long>(x1 + (198L*x0))];
                        auto tmp30 = out_ptr1[static_cast<long>(x1 + (198L*x0))];
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
                        auto tmp9 = static_cast<int>(2);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(198);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = masked_load(in_ptr2 + static_cast<long>(x2 + (768L*(static_cast<long>(((-2L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp15));
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        auto tmp26 = tmp24 + tmp25;
                        auto tmp28 = at::vec::Vectorized<float>(tmp27);
                        auto tmp29 = tmp26 - tmp28;
                        auto tmp31 = static_cast<float>(768.0);
                        auto tmp32 = tmp30 / tmp31;
                        auto tmp33 = static_cast<float>(1e-06);
                        auto tmp34 = decltype(tmp32)(tmp32 + tmp33);
                        auto tmp35 = 1 / std::sqrt(tmp34);
                        auto tmp36 = at::vec::Vectorized<float>(tmp35);
                        auto tmp37 = tmp29 * tmp36;
                        tmp37.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (152064L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_view_2 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(198L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp27 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (152064L*x0)));
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
                        auto tmp9 = static_cast<int>(2);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = tmp8 & tmp10;
                        auto tmp12 = [&]
                        {
                            auto tmp13 = masked_load(in_ptr1 + static_cast<long>(x2), to_float_mask(tmp11));
                            return tmp13;
                        }
                        ;
                        auto tmp14 = decltype(tmp12())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp12(), to_float_mask(tmp11));
                        auto tmp15 = tmp0 >= tmp9;
                        auto tmp16 = static_cast<int>(198);
                        auto tmp17 = tmp0 < tmp16;
                        auto tmp18 = [&]
                        {
                            auto tmp19 = masked_load(in_ptr2 + static_cast<long>(x2 + (768L*(static_cast<long>(((-2L) + x1)) % static_cast<long>(196L))) + (150528L*x0)), to_float_mask(tmp15));
                            return tmp19;
                        }
                        ;
                        auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp15));
                        auto tmp21 = to_float_mask(tmp11);
                        auto tmp22 = decltype(tmp14)::blendv(tmp20, tmp14, tmp21);
                        auto tmp23 = to_float_mask(tmp4);
                        auto tmp24 = decltype(tmp7)::blendv(tmp22, tmp7, tmp23);
                        auto tmp26 = tmp24 + tmp25;
                        auto tmp28 = tmp26 + tmp27;
                        tmp28.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (152064L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1216512L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp7 = static_cast<float>(1e-06);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4866048L); x0+=static_cast<long>(8L))
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_detach_div_native_layer_norm_native_layer_norm_backward_38 = async_compile.cpp('''
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
                       float* out_ptr11)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp3 = static_cast<float>(2.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 / tmp4;
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(768.0);
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr0[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr1[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr3[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr2[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr3[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr5[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr4[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr6[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr5[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr7[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr6[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr8[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr7[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr9[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr8[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr10[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr9[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr10 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr11[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr10[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr11 + static_cast<long>(x1 + (12L*x2) + (768L*x0)), static_cast<long>(12L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr12[static_cast<long>(x2 + (64L*x1) + (768L*x0))];
                        out_ptr11[static_cast<long>(x1 + (12L*x2) + (768L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr25 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156 = args
    args.clear()
    assert_size_stride(primals_1, (1, 198, 768), (152064, 768, 1))
    assert_size_stride(primals_2, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_3, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_4, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (2304, 768), (768, 1))
    assert_size_stride(primals_9, (2304, ), (1, ))
    assert_size_stride(primals_10, (768, 768), (768, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (3072, 768), (768, 1))
    assert_size_stride(primals_15, (3072, ), (1, ))
    assert_size_stride(primals_16, (768, 3072), (3072, 1))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (2304, 768), (768, 1))
    assert_size_stride(primals_21, (2304, ), (1, ))
    assert_size_stride(primals_22, (768, 768), (768, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (3072, 768), (768, 1))
    assert_size_stride(primals_27, (3072, ), (1, ))
    assert_size_stride(primals_28, (768, 3072), (3072, 1))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (2304, 768), (768, 1))
    assert_size_stride(primals_33, (2304, ), (1, ))
    assert_size_stride(primals_34, (768, 768), (768, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (3072, 768), (768, 1))
    assert_size_stride(primals_39, (3072, ), (1, ))
    assert_size_stride(primals_40, (768, 3072), (3072, 1))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (2304, 768), (768, 1))
    assert_size_stride(primals_45, (2304, ), (1, ))
    assert_size_stride(primals_46, (768, 768), (768, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (3072, 768), (768, 1))
    assert_size_stride(primals_51, (3072, ), (1, ))
    assert_size_stride(primals_52, (768, 3072), (3072, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (2304, 768), (768, 1))
    assert_size_stride(primals_57, (2304, ), (1, ))
    assert_size_stride(primals_58, (768, 768), (768, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (3072, 768), (768, 1))
    assert_size_stride(primals_63, (3072, ), (1, ))
    assert_size_stride(primals_64, (768, 3072), (3072, 1))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (2304, 768), (768, 1))
    assert_size_stride(primals_69, (2304, ), (1, ))
    assert_size_stride(primals_70, (768, 768), (768, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (3072, 768), (768, 1))
    assert_size_stride(primals_75, (3072, ), (1, ))
    assert_size_stride(primals_76, (768, 3072), (3072, 1))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (2304, 768), (768, 1))
    assert_size_stride(primals_81, (2304, ), (1, ))
    assert_size_stride(primals_82, (768, 768), (768, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (768, ), (1, ))
    assert_size_stride(primals_86, (3072, 768), (768, 1))
    assert_size_stride(primals_87, (3072, ), (1, ))
    assert_size_stride(primals_88, (768, 3072), (3072, 1))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_92, (2304, 768), (768, 1))
    assert_size_stride(primals_93, (2304, ), (1, ))
    assert_size_stride(primals_94, (768, 768), (768, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (768, ), (1, ))
    assert_size_stride(primals_97, (768, ), (1, ))
    assert_size_stride(primals_98, (3072, 768), (768, 1))
    assert_size_stride(primals_99, (3072, ), (1, ))
    assert_size_stride(primals_100, (768, 3072), (3072, 1))
    assert_size_stride(primals_101, (768, ), (1, ))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_104, (2304, 768), (768, 1))
    assert_size_stride(primals_105, (2304, ), (1, ))
    assert_size_stride(primals_106, (768, 768), (768, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (768, ), (1, ))
    assert_size_stride(primals_110, (3072, 768), (768, 1))
    assert_size_stride(primals_111, (3072, ), (1, ))
    assert_size_stride(primals_112, (768, 3072), (3072, 1))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (2304, 768), (768, 1))
    assert_size_stride(primals_117, (2304, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (3072, 768), (768, 1))
    assert_size_stride(primals_123, (3072, ), (1, ))
    assert_size_stride(primals_124, (768, 3072), (3072, 1))
    assert_size_stride(primals_125, (768, ), (1, ))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_128, (2304, 768), (768, 1))
    assert_size_stride(primals_129, (2304, ), (1, ))
    assert_size_stride(primals_130, (768, 768), (768, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, ), (1, ))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (3072, 768), (768, 1))
    assert_size_stride(primals_135, (3072, ), (1, ))
    assert_size_stride(primals_136, (768, 3072), (3072, 1))
    assert_size_stride(primals_137, (768, ), (1, ))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_140, (2304, 768), (768, 1))
    assert_size_stride(primals_141, (2304, ), (1, ))
    assert_size_stride(primals_142, (768, 768), (768, 1))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, ), (1, ))
    assert_size_stride(primals_146, (3072, 768), (768, 1))
    assert_size_stride(primals_147, (3072, ), (1, ))
    assert_size_stride(primals_148, (768, 3072), (3072, 1))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_152, (1000, 768), (768, 1))
    assert_size_stride(primals_153, (1000, ), (1, ))
    assert_size_stride(primals_154, (1000, 768), (768, 1))
    assert_size_stride(primals_155, (1000, ), (1, ))
    assert_size_stride(primals_156, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_4.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_156
    del primals_4
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_5, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 14, 14), (150528, 1, 10752, 768))
    del primals_5
    buf3 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf7 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_1(c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_7
    buf8 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___0___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_9, buf7, reinterpret_tensor(primals_8, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf8)
    del primals_9
    # Source Nodes: [x_9], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf9 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf8, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf8, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf8, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf10 = buf9[0]
    buf11 = buf9[1]
    buf12 = buf9[2]
    buf13 = buf9[3]
    buf14 = buf9[6]
    buf15 = buf9[7]
    del buf9
    buf17 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_11, reinterpret_tensor(buf10, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf17)
    del primals_11
    buf18 = reinterpret_tensor(buf17, (8, 198, 768), (152064, 768, 1), 0); del buf17  # reuse
    buf19 = buf3; del buf3  # reuse
    buf20 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf22 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf23 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_view_2(c_void_p(buf18.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del buf2
    del primals_1
    del primals_13
    del primals_2
    del primals_3
    buf24 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_14], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_15, buf23, reinterpret_tensor(primals_14, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf24)
    del primals_15
    buf25 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_3(c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    buf26 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_18], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_17, buf25, reinterpret_tensor(primals_16, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf26)
    del primals_17
    buf27 = buf19; del buf19  # reuse
    buf28 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf30 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf31 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_4(c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del primals_19
    buf32 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___1___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_21, buf31, reinterpret_tensor(primals_20, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf32)
    del primals_21
    # Source Nodes: [x_21], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf33 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf32, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf32, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf32, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf34 = buf33[0]
    buf35 = buf33[1]
    buf36 = buf33[2]
    buf37 = buf33[3]
    buf38 = buf33[6]
    buf39 = buf33[7]
    del buf33
    buf41 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_23], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_23, reinterpret_tensor(buf34, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_22, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf41)
    del primals_23
    buf42 = buf27; del buf27  # reuse
    buf43 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf45 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf46 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_5(c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del primals_25
    buf47 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_27, buf46, reinterpret_tensor(primals_26, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf47)
    del primals_27
    buf48 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_6(c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    buf49 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_29, buf48, reinterpret_tensor(primals_28, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf49)
    del primals_29
    buf50 = buf42; del buf42  # reuse
    buf51 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf53 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf54 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del primals_31
    buf55 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___2___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_33, buf54, reinterpret_tensor(primals_32, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf55)
    del primals_33
    # Source Nodes: [x_33], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf56 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf55, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf55, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf55, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf57 = buf56[0]
    buf58 = buf56[1]
    buf59 = buf56[2]
    buf60 = buf56[3]
    buf61 = buf56[6]
    buf62 = buf56[7]
    del buf56
    buf64 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_35], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_35, reinterpret_tensor(buf57, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_34, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf64)
    del primals_35
    buf65 = reinterpret_tensor(buf64, (8, 198, 768), (152064, 768, 1), 0); del buf64  # reuse
    buf66 = buf50; del buf50  # reuse
    buf67 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf69 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf70 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_8(c_void_p(buf65.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_37
    buf71 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_38], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_39, buf70, reinterpret_tensor(primals_38, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf71)
    del primals_39
    buf72 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_9(c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = buf49; del buf49  # reuse
    # Source Nodes: [x_42], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_41, buf72, reinterpret_tensor(primals_40, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf73)
    del primals_41
    buf74 = buf66; del buf66  # reuse
    buf75 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf77 = reinterpret_tensor(buf41, (8, 198, 768), (152064, 768, 1), 0); del buf41  # reuse
    buf78 = buf26; del buf26  # reuse
    cpp_fused_add_native_layer_norm_view_10(c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del primals_43
    buf79 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___3___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_45, buf78, reinterpret_tensor(primals_44, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf79)
    del primals_45
    # Source Nodes: [x_45], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf80 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf79, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf79, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf79, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf81 = buf80[0]
    buf82 = buf80[1]
    buf83 = buf80[2]
    buf84 = buf80[3]
    buf85 = buf80[6]
    buf86 = buf80[7]
    del buf80
    buf88 = reinterpret_tensor(buf18, (1584, 768), (768, 1), 0); del buf18  # reuse
    # Source Nodes: [x_47], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_47, reinterpret_tensor(buf81, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_46, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf88)
    del primals_47
    buf89 = buf74; del buf74  # reuse
    buf90 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf92 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf93 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    del primals_49
    buf94 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_51, buf93, reinterpret_tensor(primals_50, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf94)
    del primals_51
    buf95 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_12(c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    buf96 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_53, buf95, reinterpret_tensor(primals_52, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf96)
    del primals_53
    buf97 = buf89; del buf89  # reuse
    buf98 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf100 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf101 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_13(c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del primals_55
    buf102 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___4___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_57, buf101, reinterpret_tensor(primals_56, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf102)
    del primals_57
    # Source Nodes: [x_57], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf103 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf102, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf102, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf102, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf104 = buf103[0]
    buf105 = buf103[1]
    buf106 = buf103[2]
    buf107 = buf103[3]
    buf108 = buf103[6]
    buf109 = buf103[7]
    del buf103
    buf111 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_59, reinterpret_tensor(buf104, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf111)
    del primals_59
    buf112 = reinterpret_tensor(buf111, (8, 198, 768), (152064, 768, 1), 0); del buf111  # reuse
    buf113 = buf97; del buf97  # reuse
    buf114 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf116 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf117 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_14(c_void_p(buf112.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del primals_61
    buf118 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_63, buf117, reinterpret_tensor(primals_62, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf118)
    del primals_63
    buf119 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_15(c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()))
    buf120 = buf96; del buf96  # reuse
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_65, buf119, reinterpret_tensor(primals_64, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf120)
    del primals_65
    buf121 = buf113; del buf113  # reuse
    buf122 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf124 = reinterpret_tensor(buf88, (8, 198, 768), (152064, 768, 1), 0); del buf88  # reuse
    buf125 = buf73; del buf73  # reuse
    cpp_fused_add_native_layer_norm_view_16(c_void_p(buf112.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    del primals_67
    buf126 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___5___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_69, buf125, reinterpret_tensor(primals_68, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf126)
    del primals_69
    # Source Nodes: [x_69], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf127 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf126, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf126, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf126, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf128 = buf127[0]
    buf129 = buf127[1]
    buf130 = buf127[2]
    buf131 = buf127[3]
    buf132 = buf127[6]
    buf133 = buf127[7]
    del buf127
    buf135 = reinterpret_tensor(buf65, (1584, 768), (768, 1), 0); del buf65  # reuse
    # Source Nodes: [x_71], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_71, reinterpret_tensor(buf128, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_70, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf135)
    del primals_71
    buf136 = buf121; del buf121  # reuse
    buf137 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf139 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf140 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_17(c_void_p(buf112.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    del primals_73
    buf141 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_74], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_75, buf140, reinterpret_tensor(primals_74, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf141)
    del primals_75
    buf142 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_18(c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    buf143 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_77, buf142, reinterpret_tensor(primals_76, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf143)
    del primals_77
    buf144 = buf136; del buf136  # reuse
    buf145 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf147 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf148 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_19(c_void_p(buf112.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()))
    del primals_79
    buf149 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___6___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_81, buf148, reinterpret_tensor(primals_80, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf149)
    del primals_81
    # Source Nodes: [x_81], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf150 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf149, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf149, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf149, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf151 = buf150[0]
    buf152 = buf150[1]
    buf153 = buf150[2]
    buf154 = buf150[3]
    buf155 = buf150[6]
    buf156 = buf150[7]
    del buf150
    buf158 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_83], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_83, reinterpret_tensor(buf151, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_82, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf158)
    del primals_83
    buf159 = reinterpret_tensor(buf158, (8, 198, 768), (152064, 768, 1), 0); del buf158  # reuse
    buf160 = buf144; del buf144  # reuse
    buf161 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf163 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf164 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_20(c_void_p(buf159.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    del primals_85
    buf165 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_87, buf164, reinterpret_tensor(primals_86, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf165)
    del primals_87
    buf166 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_21(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = buf143; del buf143  # reuse
    # Source Nodes: [x_90], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_89, buf166, reinterpret_tensor(primals_88, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf167)
    del primals_89
    buf168 = buf160; del buf160  # reuse
    buf169 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf171 = reinterpret_tensor(buf135, (8, 198, 768), (152064, 768, 1), 0); del buf135  # reuse
    buf172 = buf120; del buf120  # reuse
    cpp_fused_add_native_layer_norm_view_22(c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del primals_91
    buf173 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___7___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_93, buf172, reinterpret_tensor(primals_92, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf173)
    del primals_93
    # Source Nodes: [x_93], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf174 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf173, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf173, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf173, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf175 = buf174[0]
    buf176 = buf174[1]
    buf177 = buf174[2]
    buf178 = buf174[3]
    buf179 = buf174[6]
    buf180 = buf174[7]
    del buf174
    buf182 = reinterpret_tensor(buf112, (1584, 768), (768, 1), 0); del buf112  # reuse
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_95, reinterpret_tensor(buf175, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_94, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf182)
    del primals_95
    buf183 = buf168; del buf168  # reuse
    buf184 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf186 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf187 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()))
    del primals_97
    buf188 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_98], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_99, buf187, reinterpret_tensor(primals_98, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf188)
    del primals_99
    buf189 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_24(c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_102], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_101, buf189, reinterpret_tensor(primals_100, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf190)
    del primals_101
    buf191 = buf183; del buf183  # reuse
    buf192 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf194 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf195 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()))
    del primals_103
    buf196 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___8___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_105, buf195, reinterpret_tensor(primals_104, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf196)
    del primals_105
    # Source Nodes: [x_105], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf197 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf198 = buf197[0]
    buf199 = buf197[1]
    buf200 = buf197[2]
    buf201 = buf197[3]
    buf202 = buf197[6]
    buf203 = buf197[7]
    del buf197
    buf205 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_107, reinterpret_tensor(buf198, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf205)
    del primals_107
    buf206 = reinterpret_tensor(buf205, (8, 198, 768), (152064, 768, 1), 0); del buf205  # reuse
    buf207 = buf191; del buf191  # reuse
    buf208 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf210 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf211 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_26(c_void_p(buf206.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_109
    buf212 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_110], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf211, reinterpret_tensor(primals_110, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf212)
    del primals_111
    buf213 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_27(c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = buf190; del buf190  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_113, buf213, reinterpret_tensor(primals_112, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf214)
    del primals_113
    buf215 = buf207; del buf207  # reuse
    buf216 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf218 = reinterpret_tensor(buf182, (8, 198, 768), (152064, 768, 1), 0); del buf182  # reuse
    buf219 = buf167; del buf167  # reuse
    cpp_fused_add_native_layer_norm_view_28(c_void_p(buf206.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    del primals_115
    buf220 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___9___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_117, buf219, reinterpret_tensor(primals_116, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf220)
    del primals_117
    # Source Nodes: [x_117], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf221 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf220, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf220, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf220, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf222 = buf221[0]
    buf223 = buf221[1]
    buf224 = buf221[2]
    buf225 = buf221[3]
    buf226 = buf221[6]
    buf227 = buf221[7]
    del buf221
    buf229 = reinterpret_tensor(buf159, (1584, 768), (768, 1), 0); del buf159  # reuse
    # Source Nodes: [x_119], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_119, reinterpret_tensor(buf222, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf229)
    del primals_119
    buf230 = buf215; del buf215  # reuse
    buf231 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf233 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf234 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_29(c_void_p(buf206.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    del primals_121
    buf235 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_123, buf234, reinterpret_tensor(primals_122, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf235)
    del primals_123
    buf236 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_30(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_126], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_125, buf236, reinterpret_tensor(primals_124, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf237)
    del primals_125
    buf238 = buf230; del buf230  # reuse
    buf239 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf241 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf242 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_31(c_void_p(buf206.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del primals_127
    buf243 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___10___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_129, buf242, reinterpret_tensor(primals_128, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf243)
    del primals_129
    # Source Nodes: [x_129], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf244 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf243, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf243, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf243, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf245 = buf244[0]
    buf246 = buf244[1]
    buf247 = buf244[2]
    buf248 = buf244[3]
    buf249 = buf244[6]
    buf250 = buf244[7]
    del buf244
    buf252 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_131, reinterpret_tensor(buf245, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_130, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf252)
    del primals_131
    buf253 = reinterpret_tensor(buf252, (8, 198, 768), (152064, 768, 1), 0); del buf252  # reuse
    buf254 = buf238; del buf238  # reuse
    buf255 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf257 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf258 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_32(c_void_p(buf253.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()))
    del primals_133
    buf259 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_135, buf258, reinterpret_tensor(primals_134, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf259)
    del primals_135
    buf260 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_33(c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = buf237; del buf237  # reuse
    # Source Nodes: [x_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_137, buf260, reinterpret_tensor(primals_136, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf261)
    del primals_137
    buf262 = buf254; del buf254  # reuse
    buf263 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf265 = reinterpret_tensor(buf229, (8, 198, 768), (152064, 768, 1), 0); del buf229  # reuse
    buf266 = buf214; del buf214  # reuse
    cpp_fused_add_native_layer_norm_view_34(c_void_p(buf253.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    del primals_139
    buf267 = empty((1584, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___blocks___11___attn_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_141, buf266, reinterpret_tensor(primals_140, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf267)
    del primals_141
    # Source Nodes: [x_141], Original ATen: [aten._scaled_dot_product_flash_attention]
    buf268 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf267, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf267, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf267, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536))
    buf269 = buf268[0]
    buf270 = buf268[1]
    buf271 = buf268[2]
    buf272 = buf268[3]
    buf273 = buf268[6]
    buf274 = buf268[7]
    del buf268
    buf276 = reinterpret_tensor(buf206, (1584, 768), (768, 1), 0); del buf206  # reuse
    # Source Nodes: [x_143], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_143, reinterpret_tensor(buf269, (1584, 768), (768, 1), 0), reinterpret_tensor(primals_142, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf276)
    del primals_143
    buf277 = buf262; del buf262  # reuse
    buf278 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf280 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf281 = empty((1584, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf253.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del primals_145
    buf282 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_146], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_147, buf281, reinterpret_tensor(primals_146, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf282)
    del primals_147
    buf283 = empty((1584, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_36(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    buf284 = empty((1584, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_150], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf283, reinterpret_tensor(primals_148, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf284)
    del primals_149
    buf285 = buf277; del buf277  # reuse
    buf286 = empty_strided((8, 198, 1), (198, 1, 1584), device='cpu', dtype=torch.float32)
    buf288 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    buf289 = empty((8, 198, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_37(c_void_p(buf253.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    del buf285
    del primals_151
    buf290 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_153, reinterpret_tensor(buf289, (8, 768), (152064, 1), 0), reinterpret_tensor(primals_152, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf290)
    del primals_153
    buf291 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_dist_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_155, reinterpret_tensor(buf289, (8, 768), (152064, 1), 768), reinterpret_tensor(primals_154, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf291)
    del primals_155
    buf292 = buf290; del buf290  # reuse
    buf293 = reinterpret_tensor(buf286, (8, 198, 1), (198, 1, 1), 0); del buf286  # reuse
    buf294 = reinterpret_tensor(buf278, (8, 198, 1), (198, 1, 1), 0); del buf278  # reuse
    buf295 = reinterpret_tensor(buf284, (8, 12, 198, 64), (152064, 1, 768, 12), 0); del buf284  # reuse
    buf296 = reinterpret_tensor(buf263, (8, 198, 1), (198, 1, 1), 0); del buf263  # reuse
    buf297 = reinterpret_tensor(buf255, (8, 198, 1), (198, 1, 1), 0); del buf255  # reuse
    buf298 = reinterpret_tensor(buf276, (8, 12, 198, 64), (152064, 1, 768, 12), 0); del buf276  # reuse
    buf299 = reinterpret_tensor(buf239, (8, 198, 1), (198, 1, 1), 0); del buf239  # reuse
    buf300 = reinterpret_tensor(buf231, (8, 198, 1), (198, 1, 1), 0); del buf231  # reuse
    buf301 = reinterpret_tensor(buf261, (8, 12, 198, 64), (152064, 1, 768, 12), 0); del buf261  # reuse
    buf302 = reinterpret_tensor(buf216, (8, 198, 1), (198, 1, 1), 0); del buf216  # reuse
    buf303 = reinterpret_tensor(buf208, (8, 198, 1), (198, 1, 1), 0); del buf208  # reuse
    buf304 = reinterpret_tensor(buf253, (8, 12, 198, 64), (152064, 1, 768, 12), 0); del buf253  # reuse
    buf305 = reinterpret_tensor(buf192, (8, 198, 1), (198, 1, 1), 0); del buf192  # reuse
    buf306 = reinterpret_tensor(buf184, (8, 198, 1), (198, 1, 1), 0); del buf184  # reuse
    buf307 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf308 = reinterpret_tensor(buf169, (8, 198, 1), (198, 1, 1), 0); del buf169  # reuse
    buf309 = reinterpret_tensor(buf161, (8, 198, 1), (198, 1, 1), 0); del buf161  # reuse
    buf310 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf311 = reinterpret_tensor(buf145, (8, 198, 1), (198, 1, 1), 0); del buf145  # reuse
    buf312 = reinterpret_tensor(buf137, (8, 198, 1), (198, 1, 1), 0); del buf137  # reuse
    buf313 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf314 = reinterpret_tensor(buf122, (8, 198, 1), (198, 1, 1), 0); del buf122  # reuse
    buf315 = reinterpret_tensor(buf114, (8, 198, 1), (198, 1, 1), 0); del buf114  # reuse
    buf316 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf317 = reinterpret_tensor(buf98, (8, 198, 1), (198, 1, 1), 0); del buf98  # reuse
    buf318 = reinterpret_tensor(buf90, (8, 198, 1), (198, 1, 1), 0); del buf90  # reuse
    buf319 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf320 = reinterpret_tensor(buf75, (8, 198, 1), (198, 1, 1), 0); del buf75  # reuse
    buf321 = reinterpret_tensor(buf67, (8, 198, 1), (198, 1, 1), 0); del buf67  # reuse
    buf322 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf323 = reinterpret_tensor(buf51, (8, 198, 1), (198, 1, 1), 0); del buf51  # reuse
    buf324 = reinterpret_tensor(buf43, (8, 198, 1), (198, 1, 1), 0); del buf43  # reuse
    buf325 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf326 = reinterpret_tensor(buf28, (8, 198, 1), (198, 1, 1), 0); del buf28  # reuse
    buf327 = reinterpret_tensor(buf20, (8, 198, 1), (198, 1, 1), 0); del buf20  # reuse
    buf328 = empty_strided((8, 12, 198, 64), (152064, 1, 768, 12), device='cpu', dtype=torch.float32)
    buf329 = reinterpret_tensor(buf4, (8, 198, 1), (198, 1, 1), 0); del buf4  # reuse
    cpp_fused_add_cat_detach_div_native_layer_norm_native_layer_norm_backward_38(c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf328.data_ptr()))
    return (buf292, buf0, primals_6, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, buf1, buf6, buf7, reinterpret_tensor(buf8, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf8, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf8, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf11, buf12, buf13, buf14, buf15, reinterpret_tensor(buf10, (1584, 768), (768, 1), 0), buf22, buf23, buf24, buf25, buf30, buf31, reinterpret_tensor(buf32, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf32, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf32, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf35, buf36, buf37, buf38, buf39, reinterpret_tensor(buf34, (1584, 768), (768, 1), 0), buf45, buf46, buf47, buf48, buf53, buf54, reinterpret_tensor(buf55, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf55, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf55, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf58, buf59, buf60, buf61, buf62, reinterpret_tensor(buf57, (1584, 768), (768, 1), 0), buf69, buf70, buf71, buf72, buf77, buf78, reinterpret_tensor(buf79, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf79, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf79, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf82, buf83, buf84, buf85, buf86, reinterpret_tensor(buf81, (1584, 768), (768, 1), 0), buf92, buf93, buf94, buf95, buf100, buf101, reinterpret_tensor(buf102, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf102, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf102, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf105, buf106, buf107, buf108, buf109, reinterpret_tensor(buf104, (1584, 768), (768, 1), 0), buf116, buf117, buf118, buf119, buf124, buf125, reinterpret_tensor(buf126, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf126, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf126, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf129, buf130, buf131, buf132, buf133, reinterpret_tensor(buf128, (1584, 768), (768, 1), 0), buf139, buf140, buf141, buf142, buf147, buf148, reinterpret_tensor(buf149, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf149, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf149, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf152, buf153, buf154, buf155, buf156, reinterpret_tensor(buf151, (1584, 768), (768, 1), 0), buf163, buf164, buf165, buf166, buf171, buf172, reinterpret_tensor(buf173, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf173, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf173, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf176, buf177, buf178, buf179, buf180, reinterpret_tensor(buf175, (1584, 768), (768, 1), 0), buf186, buf187, buf188, buf189, buf194, buf195, reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf196, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf199, buf200, buf201, buf202, buf203, reinterpret_tensor(buf198, (1584, 768), (768, 1), 0), buf210, buf211, buf212, buf213, buf218, buf219, reinterpret_tensor(buf220, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf220, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf220, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf223, buf224, buf225, buf226, buf227, reinterpret_tensor(buf222, (1584, 768), (768, 1), 0), buf233, buf234, buf235, buf236, buf241, buf242, reinterpret_tensor(buf243, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf243, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf243, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf246, buf247, buf248, buf249, buf250, reinterpret_tensor(buf245, (1584, 768), (768, 1), 0), buf257, buf258, buf259, buf260, buf265, buf266, reinterpret_tensor(buf267, (8, 12, 198, 64), (456192, 64, 2304, 1), 0), reinterpret_tensor(buf267, (8, 12, 198, 64), (456192, 64, 2304, 1), 768), reinterpret_tensor(buf267, (8, 12, 198, 64), (456192, 64, 2304, 1), 1536), buf270, buf271, buf272, buf273, buf274, reinterpret_tensor(buf269, (1584, 768), (768, 1), 0), buf280, buf281, buf282, buf283, buf288, reinterpret_tensor(buf289, (8, 768), (152064, 1), 0), reinterpret_tensor(buf289, (8, 768), (152064, 1), 768), reinterpret_tensor(primals_154, (1000, 768), (768, 1), 0), reinterpret_tensor(primals_152, (1000, 768), (768, 1), 0), buf293, reinterpret_tensor(primals_148, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_146, (3072, 768), (768, 1), 0), buf294, reinterpret_tensor(primals_142, (768, 768), (768, 1), 0), buf295, reinterpret_tensor(primals_140, (2304, 768), (768, 1), 0), buf296, reinterpret_tensor(primals_136, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_134, (3072, 768), (768, 1), 0), buf297, reinterpret_tensor(primals_130, (768, 768), (768, 1), 0), buf298, reinterpret_tensor(primals_128, (2304, 768), (768, 1), 0), buf299, reinterpret_tensor(primals_124, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_122, (3072, 768), (768, 1), 0), buf300, reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), buf301, reinterpret_tensor(primals_116, (2304, 768), (768, 1), 0), buf302, reinterpret_tensor(primals_112, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_110, (3072, 768), (768, 1), 0), buf303, reinterpret_tensor(primals_106, (768, 768), (768, 1), 0), buf304, reinterpret_tensor(primals_104, (2304, 768), (768, 1), 0), buf305, reinterpret_tensor(primals_100, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_98, (3072, 768), (768, 1), 0), buf306, reinterpret_tensor(primals_94, (768, 768), (768, 1), 0), buf307, reinterpret_tensor(primals_92, (2304, 768), (768, 1), 0), buf308, reinterpret_tensor(primals_88, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_86, (3072, 768), (768, 1), 0), buf309, reinterpret_tensor(primals_82, (768, 768), (768, 1), 0), buf310, reinterpret_tensor(primals_80, (2304, 768), (768, 1), 0), buf311, reinterpret_tensor(primals_76, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_74, (3072, 768), (768, 1), 0), buf312, reinterpret_tensor(primals_70, (768, 768), (768, 1), 0), buf313, reinterpret_tensor(primals_68, (2304, 768), (768, 1), 0), buf314, reinterpret_tensor(primals_64, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_62, (3072, 768), (768, 1), 0), buf315, reinterpret_tensor(primals_58, (768, 768), (768, 1), 0), buf316, reinterpret_tensor(primals_56, (2304, 768), (768, 1), 0), buf317, reinterpret_tensor(primals_52, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_50, (3072, 768), (768, 1), 0), buf318, reinterpret_tensor(primals_46, (768, 768), (768, 1), 0), buf319, reinterpret_tensor(primals_44, (2304, 768), (768, 1), 0), buf320, reinterpret_tensor(primals_40, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_38, (3072, 768), (768, 1), 0), buf321, reinterpret_tensor(primals_34, (768, 768), (768, 1), 0), buf322, reinterpret_tensor(primals_32, (2304, 768), (768, 1), 0), buf323, reinterpret_tensor(primals_28, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_26, (3072, 768), (768, 1), 0), buf324, reinterpret_tensor(primals_22, (768, 768), (768, 1), 0), buf325, reinterpret_tensor(primals_20, (2304, 768), (768, 1), 0), buf326, reinterpret_tensor(primals_16, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_14, (3072, 768), (768, 1), 0), buf327, reinterpret_tensor(primals_10, (768, 768), (768, 1), 0), buf328, reinterpret_tensor(primals_8, (2304, 768), (768, 1), 0), buf329, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 198, 768), (152064, 768, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 768), (768, 768, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1, 1, 768), (768, 768, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((2304, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('deit_base_distilled_patch16_224', benchmark_compiled_module)
