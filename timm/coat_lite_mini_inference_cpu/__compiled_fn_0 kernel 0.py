
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (16L*x1) + (48L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (48L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_native_layer_norm_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x1), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x1 + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = out_ptr0[static_cast<long>((-1L) + x2 + (3136L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp16 = out_ptr1[static_cast<long>((-1L) + x2 + (3136L*x0))];
                            auto tmp17 = static_cast<float>(64.0);
                            auto tmp18 = tmp16 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = 1 / std::sqrt(tmp20);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp15 * tmp22;
                            auto tmp24 = masked_load(in_ptr2 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp25 = tmp23 * tmp24;
                            auto tmp26 = masked_load(in_ptr3 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp27 = tmp25 + tmp26;
                            return tmp27;
                        }
                        ;
                        auto tmp28 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp29 = to_float_mask(tmp4);
                        auto tmp30 = decltype(tmp7)::blendv(tmp28, tmp7, tmp29);
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp30.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (3137L*x1) + (3137L*x1_inner) + (200768L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(1L + x2 + (3137L*x1) + (3137L*x1_inner) + (200768L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (200704L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_2 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((3137L*x2) + (200768L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(3137);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (3137L*x2) + (200768L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((3137L*x2) + (200768L*x0))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0))];
                                auto tmp22 = in_ptr0[static_cast<long>(1L + (3137L*x2) + (200768L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L)))];
                                auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp25 = tmp4 ? tmp19 : tmp24;
                            tmp_acc0 = welford_combine(tmp_acc0, tmp16);
                            tmp_acc1 = welford_combine(tmp_acc1, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (3137L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (3137L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr0[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp19 = out_ptr1[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp26 = in_ptr2[static_cast<long>(x2)];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((3137L*x2) + (200768L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (3137L*x2) + (200768L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(64.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                        out_ptr2[static_cast<long>(x2 + (64L*x1) + (200768L*x0))] = tmp29;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x1 + (192L*x2) + (602304L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (8L*x1) + (64L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
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
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0))];
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (8L*x1) + (192L*x2) + (602304L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (8L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (8L*x1) + (16L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (50176L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(40);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-16L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(64);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-40L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_6 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp17 = in_ptr2[static_cast<long>(x2 + (64L*x1) + (200768L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((3137L*x2) + (200768L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(3137);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (3137L*x2) + (200768L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((3137L*x2) + (200768L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp4 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr1[static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0))];
                                auto tmp24 = in_ptr0[static_cast<long>(1L + (3137L*x2) + (200768L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L)))];
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
                        out_ptr0[static_cast<long>(x1 + (3137L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (3137L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = in_ptr2[static_cast<long>(x2 + (64L*x1) + (200768L*x0))];
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp30 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((3137L*x2) + (200768L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (3137L*x2) + (200768L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                        auto tmp22 = static_cast<float>(64.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                        out_ptr2[static_cast<long>(x2 + (64L*x1) + (200768L*x0))] = tmp31;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12849152L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((3137L*x2) + (3137L*x2_inner) + (200768L*x0))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>(1L + (3137L*x2) + (3137L*x2_inner) + (200768L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L)))]; return masked_load(tmpbuf, to_float_mask(tmp8)); })();
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_9 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(3137);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (3137L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (3137L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x2 + (3137L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x2 + (3137L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (200768L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(64L + x1 + (64L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(64.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x1 + (192L*x2) + (602304L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x2 + (192L*x1) + (602304L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (64L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (200768L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (8L*x1) + (64L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (8L*x1) + (192L*x2) + (602304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3137L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (8L*x2) + (25096L*x1) + (200768L*x0))];
                            auto tmp1 = static_cast<float>(0.3535533905932738);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (8L*x1) + (192L*x2) + (602304L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (8L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(16);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (8L*x1) + (16L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (50176L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(40);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-16L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(64);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-40L) + x3 + (8L*x1) + (24L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(3136L))) + (75264L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (8L*x1) + (64L*x2) + (200768L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_13 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(3137);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (3137L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (3137L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3137L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (3137L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(3136L))) + (200768L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(64.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (200768L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12849152L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_convolution_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (200768L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(3137);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(64L + x2 + (64L*x1) + (200768L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (256L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_native_layer_norm_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x1), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x1 + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = out_ptr0[static_cast<long>((-1L) + x2 + (784L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp16 = out_ptr1[static_cast<long>((-1L) + x2 + (784L*x0))];
                            auto tmp17 = static_cast<float>(128.0);
                            auto tmp18 = tmp16 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = 1 / std::sqrt(tmp20);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp15 * tmp22;
                            auto tmp24 = masked_load(in_ptr2 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp25 = tmp23 * tmp24;
                            auto tmp26 = masked_load(in_ptr3 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp27 = tmp25 + tmp26;
                            return tmp27;
                        }
                        ;
                        auto tmp28 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp29 = to_float_mask(tmp4);
                        auto tmp30 = decltype(tmp7)::blendv(tmp28, tmp7, tmp29);
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp30.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (785L*x1) + (785L*x1_inner) + (100480L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(1L + x2 + (785L*x1) + (785L*x1_inner) + (100480L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (128L*x2) + (100352L*x0)), static_cast<long>(128L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((785L*x2) + (100480L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(785);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (785L*x2) + (100480L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(784L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((785L*x2) + (100480L*x0))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0))];
                                auto tmp22 = in_ptr0[static_cast<long>(1L + (785L*x2) + (100480L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(784L)))];
                                auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp25 = tmp4 ? tmp19 : tmp24;
                            tmp_acc0 = welford_combine(tmp_acc0, tmp16);
                            tmp_acc1 = welford_combine(tmp_acc1, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (785L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (785L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr0[static_cast<long>(x1 + (785L*x0))];
                        auto tmp19 = out_ptr1[static_cast<long>(x1 + (785L*x0))];
                        auto tmp26 = in_ptr2[static_cast<long>(x2)];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((785L*x2) + (100480L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (785L*x2) + (100480L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(784L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(128.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                        out_ptr2[static_cast<long>(x2 + (128L*x1) + (100480L*x0))] = tmp29;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (301440L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (16L*x1) + (128L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (16L*x1) + (384L*x2) + (301440L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (16L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(32);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (16L*x1) + (32L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (25088L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(80);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-32L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(128);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-80L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_21 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                        {
                            auto tmp17 = in_ptr2[static_cast<long>(x2 + (128L*x1) + (100480L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((785L*x2) + (100480L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(785);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (785L*x2) + (100480L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(784L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((785L*x2) + (100480L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp4 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr1[static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0))];
                                auto tmp24 = in_ptr0[static_cast<long>(1L + (785L*x2) + (100480L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(784L)))];
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
                        out_ptr0[static_cast<long>(x1 + (785L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (785L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = in_ptr2[static_cast<long>(x2 + (128L*x1) + (100480L*x0))];
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (785L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (785L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp30 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((785L*x2) + (100480L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (785L*x2) + (100480L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(784L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                        auto tmp22 = static_cast<float>(128.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                        out_ptr2[static_cast<long>(x2 + (128L*x1) + (100480L*x0))] = tmp31;
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6430720L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((785L*x2) + (785L*x2_inner) + (100480L*x0))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>(1L + (785L*x2) + (785L*x2_inner) + (100480L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(784L)))]; return masked_load(tmpbuf, to_float_mask(tmp8)); })();
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(785);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (785L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (785L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x2 + (785L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x2 + (785L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (100480L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(128L + x1 + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(128.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x1 + (384L*x2) + (301440L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (384L*x1) + (301440L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (128L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (100480L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (16L*x1) + (128L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (384L*x2) + (301440L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0)));
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(785L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (16L*x2) + (12560L*x1) + (100480L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (16L*x1) + (384L*x2) + (301440L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (16L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(32);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (16L*x1) + (32L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (25088L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(80);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-32L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(128);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-80L) + x3 + (16L*x1) + (48L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(784L))) + (37632L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (16L*x1) + (128L*x2) + (100480L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_28 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(785);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (785L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (785L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(785L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (785L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (785L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(784L))) + (100480L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(128.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (100480L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6430720L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_convolution_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (100480L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(785);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(128L + x2 + (128L*x1) + (100480L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (128L*x2) + (512L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_native_layer_norm_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (320L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x1), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x1 + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = out_ptr0[static_cast<long>((-1L) + x2 + (196L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp16 = out_ptr1[static_cast<long>((-1L) + x2 + (196L*x0))];
                            auto tmp17 = static_cast<float>(320.0);
                            auto tmp18 = tmp16 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = 1 / std::sqrt(tmp20);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp15 * tmp22;
                            auto tmp24 = masked_load(in_ptr2 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp25 = tmp23 * tmp24;
                            auto tmp26 = masked_load(in_ptr3 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp27 = tmp25 + tmp26;
                            return tmp27;
                        }
                        ;
                        auto tmp28 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp29 = to_float_mask(tmp4);
                        auto tmp30 = decltype(tmp7)::blendv(tmp28, tmp7, tmp29);
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp30.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (63040L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(1L + x2 + (197L*x1) + (197L*x1_inner) + (63040L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (320L*x2) + (62720L*x0)), static_cast<long>(320L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(1L + x2 + (197L*x1) + (197L*x1_inner) + (63040L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (320L*x2) + (62720L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((197L*x2) + (63040L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (197L*x2) + (63040L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(196L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((197L*x2) + (63040L*x0))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0))];
                                auto tmp22 = in_ptr0[static_cast<long>(1L + (197L*x2) + (63040L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(196L)))];
                                auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp25 = tmp4 ? tmp19 : tmp24;
                            tmp_acc0 = welford_combine(tmp_acc0, tmp16);
                            tmp_acc1 = welford_combine(tmp_acc1, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp19 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp26 = in_ptr2[static_cast<long>(x2)];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((197L*x2) + (63040L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (197L*x2) + (63040L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(196L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(320.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                        out_ptr2[static_cast<long>(x2 + (320L*x1) + (63040L*x0))] = tmp29;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x1 + (960L*x2) + (189120L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (320L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (40L*x1) + (320L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(640L + x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0))];
                            auto tmp1 = static_cast<float>(0.15811388300841897);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (40L*x1) + (960L*x2) + (189120L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (40L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(80);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (40L*x1) + (80L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (15680L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(200);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-80L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(320);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-200L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_36 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                        {
                            auto tmp17 = in_ptr2[static_cast<long>(x2 + (320L*x1) + (63040L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((197L*x2) + (63040L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (197L*x2) + (63040L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(196L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((197L*x2) + (63040L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp4 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr1[static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0))];
                                auto tmp24 = in_ptr0[static_cast<long>(1L + (197L*x2) + (63040L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(196L)))];
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
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = in_ptr2[static_cast<long>(x2 + (320L*x1) + (63040L*x0))];
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp30 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((197L*x2) + (63040L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (197L*x2) + (63040L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(196L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                        auto tmp22 = static_cast<float>(320.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                        out_ptr2[static_cast<long>(x2 + (320L*x1) + (63040L*x0))] = tmp31;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2017280L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((197L*x2) + (197L*x2_inner) + (63040L*x0))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>(1L + (197L*x2) + (197L*x2_inner) + (63040L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(196L)))]; return masked_load(tmpbuf, to_float_mask(tmp8)); })();
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x2 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x2 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (63040L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(320L + x1 + (320L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(320.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x1 + (960L*x2) + (189120L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(320L + x2 + (960L*x1) + (189120L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (320L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (320L*x2) + (63040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (320L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (40L*x1) + (320L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(640L + x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (40L*x1) + (960L*x2) + (189120L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(40L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (40L*x2) + (7880L*x1) + (63040L*x0))];
                            auto tmp1 = static_cast<float>(0.15811388300841897);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (40L*x1) + (960L*x2) + (189120L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (40L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(80);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (40L*x1) + (80L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (15680L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(200);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-80L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(320);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-200L) + x3 + (40L*x1) + (120L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (23520L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (40L*x1) + (320L*x2) + (63040L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_43 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(197);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (63040L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(320.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (320L*x1) + (63040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2017280L); x0+=static_cast<long>(8L))
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


cpp_fused_clone_convolution_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)));
                        auto tmp0 = c10::convert<int>(1L + x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (63040L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(320L + x2 + (320L*x1) + (63040L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (320L*x1) + (62720L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (1280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (320L*x2) + (1280L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_native_layer_norm_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr1 + static_cast<long>(x1), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr0 + static_cast<long>(x1 + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = out_ptr0[static_cast<long>((-1L) + x2 + (49L*x0))];
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 - tmp14;
                            auto tmp16 = out_ptr1[static_cast<long>((-1L) + x2 + (49L*x0))];
                            auto tmp17 = static_cast<float>(512.0);
                            auto tmp18 = tmp16 / tmp17;
                            auto tmp19 = static_cast<float>(1e-05);
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = 1 / std::sqrt(tmp20);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = tmp15 * tmp22;
                            auto tmp24 = masked_load(in_ptr2 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp25 = tmp23 * tmp24;
                            auto tmp26 = masked_load(in_ptr3 + static_cast<long>(x1), to_float_mask(tmp8));
                            auto tmp27 = tmp25 + tmp26;
                            return tmp27;
                        }
                        ;
                        auto tmp28 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp29 = to_float_mask(tmp4);
                        auto tmp30 = decltype(tmp7)::blendv(tmp28, tmp7, tmp29);
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp30.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x2 + (50L*x1) + (50L*x1_inner) + (25600L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(1L + x2 + (50L*x1) + (50L*x1_inner) + (25600L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (512L*x2) + (25088L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>(1L + x2 + (50L*x1) + (50L*x1_inner) + (25600L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (512L*x2) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((50L*x2) + (25600L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(50);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (50L*x2) + (25600L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(49L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((50L*x2) + (25600L*x0))];
                                return tmp18;
                            }
                            ;
                            auto tmp19 = tmp4 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp20 = [&]
                            {
                                auto tmp21 = in_ptr1[static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0))];
                                auto tmp22 = in_ptr0[static_cast<long>(1L + (50L*x2) + (25600L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(49L)))];
                                auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                                return tmp23;
                            }
                            ;
                            auto tmp24 = tmp8 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp25 = tmp4 ? tmp19 : tmp24;
                            tmp_acc0 = welford_combine(tmp_acc0, tmp16);
                            tmp_acc1 = welford_combine(tmp_acc1, tmp25);
                        }
                        out_ptr0[static_cast<long>(x1 + (50L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (50L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = out_ptr0[static_cast<long>(x1 + (50L*x0))];
                        auto tmp19 = out_ptr1[static_cast<long>(x1 + (50L*x0))];
                        auto tmp26 = in_ptr2[static_cast<long>(x2)];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((50L*x2) + (25600L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (50L*x2) + (25600L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(49L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 - tmp17);
                        auto tmp20 = static_cast<float>(512.0);
                        auto tmp21 = tmp19 / tmp20;
                        auto tmp22 = static_cast<float>(1e-06);
                        auto tmp23 = decltype(tmp21)(tmp21 + tmp22);
                        auto tmp24 = 1 / std::sqrt(tmp23);
                        auto tmp25 = decltype(tmp18)(tmp18 * tmp24);
                        auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                        out_ptr2[static_cast<long>(x2 + (512L*x1) + (25600L*x0))] = tmp29;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (76800L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1536L*x1) + (76800L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0))];
                            auto tmp1 = static_cast<float>(0.125);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (64L*x1) + (1536L*x2) + (76800L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (64L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(128);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (64L*x1) + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (6272L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(320);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-128L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(512);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-320L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_51 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                        {
                            auto tmp17 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (25600L*x0))];
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((50L*x2) + (25600L*x0))];
                                return tmp6;
                            }
                            ;
                            auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<long>(50);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr1[static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0))];
                                auto tmp13 = in_ptr0[static_cast<long>(1L + (50L*x2) + (25600L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(49L)))];
                                auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp16 = tmp4 ? tmp7 : tmp15;
                            auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((50L*x2) + (25600L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp4 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = [&]
                            {
                                auto tmp23 = in_ptr1[static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0))];
                                auto tmp24 = in_ptr0[static_cast<long>(1L + (50L*x2) + (25600L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(49L)))];
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
                        out_ptr0[static_cast<long>(x1 + (50L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (50L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(1L))
                    {
                        auto tmp17 = in_ptr2[static_cast<long>(x2 + (512L*x1) + (25600L*x0))];
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (50L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (50L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp30 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((50L*x2) + (25600L*x0))];
                            return tmp6;
                        }
                        ;
                        auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<long>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = in_ptr1[static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0))];
                            auto tmp13 = in_ptr0[static_cast<long>(1L + (50L*x2) + (25600L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(49L)))];
                            auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                            return tmp14;
                        }
                        ;
                        auto tmp15 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                        auto tmp16 = tmp4 ? tmp7 : tmp15;
                        auto tmp18 = decltype(tmp16)(tmp16 + tmp17);
                        auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                        auto tmp22 = static_cast<float>(512.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                        out_ptr2[static_cast<long>(x2 + (512L*x1) + (25600L*x0))] = tmp31;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(819200L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((50L*x2) + (50L*x2_inner) + (25600L*x0))]; return masked_load(tmpbuf, to_float_mask(tmp4)); })();
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>(1L + (50L*x2) + (50L*x2_inner) + (25600L*x0) + (static_cast<long>(((-1L) + x1)) % static_cast<long>(49L)))]; return masked_load(tmpbuf, to_float_mask(tmp8)); })();
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_native_layer_norm_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
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
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(50);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (50L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (50L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x2 + (50L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x2 + (50L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1 + (25600L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(512L + x1 + (512L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
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
                        tmp32.store(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x1 + (1536L*x2) + (76800L*x0)));
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1536L*x1) + (76800L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (512L*x0)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp3 = tmp2.exp();
                        tmp3.store(out_ptr1 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (512L*x2) + (25600L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x3 + (64L*x1) + (512L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (1536L*x2) + (76800L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0)));
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
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (64L*x2) + (3200L*x1) + (25600L*x0))];
                            auto tmp1 = static_cast<float>(0.125);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp3 = c10::convert<long>((-1L) + x2);
                            auto tmp4 = static_cast<long>(0);
                            auto tmp5 = tmp3 >= tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr1[static_cast<long>(x3 + (64L*x1) + (1536L*x2) + (76800L*x0))];
                                auto tmp8 = c10::convert<long>(x3 + (64L*x1));
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(128);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = [&]
                                {
                                    auto tmp14 = in_ptr2[static_cast<long>(x3 + (64L*x1) + (128L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (6272L*x0))];
                                    return tmp14;
                                }
                                ;
                                auto tmp15 = tmp12 ? tmp13() : static_cast<decltype(tmp13())>(0.0);
                                auto tmp16 = tmp8 >= tmp11;
                                auto tmp17 = static_cast<long>(320);
                                auto tmp18 = tmp8 < tmp17;
                                auto tmp19 = tmp16 & tmp18;
                                auto tmp20 = [&]
                                {
                                    auto tmp21 = in_ptr3[static_cast<long>((-128L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                    return tmp21;
                                }
                                ;
                                auto tmp22 = tmp19 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                                auto tmp23 = tmp8 >= tmp17;
                                auto tmp24 = static_cast<long>(512);
                                auto tmp25 = tmp8 < tmp24;
                                auto tmp26 = [&]
                                {
                                    auto tmp27 = in_ptr4[static_cast<long>((-320L) + x3 + (64L*x1) + (192L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(49L))) + (9408L*x0))];
                                    return tmp27;
                                }
                                ;
                                auto tmp28 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                                auto tmp29 = tmp19 ? tmp22 : tmp28;
                                auto tmp30 = tmp12 ? tmp15 : tmp29;
                                auto tmp31 = decltype(tmp7)(tmp7 * tmp30);
                                return tmp31;
                            }
                            ;
                            auto tmp32 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            auto tmp33 = decltype(tmp2)(tmp2 + tmp32);
                            out_ptr0[static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25600L*x0))] = tmp33;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_58 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
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
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp4));
                                return tmp6;
                            }
                            ;
                            auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                            auto tmp8 = tmp0 >= tmp3;
                            auto tmp9 = static_cast<int>(50);
                            auto tmp10 = tmp0 < tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                                auto tmp14 = tmp12 + tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                            auto tmp16 = to_float_mask(tmp4);
                            auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (50L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (50L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (50L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (50L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
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
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(819200L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_clone_native_layer_norm_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(50L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (25600L*x0)), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(50);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25088L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr0 + static_cast<long>(512L + x2 + (512L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(49L))) + (25600L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (512L*x1) + (25600L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(400L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (25600L*x0)));
                        auto tmp1 = out_ptr0[static_cast<long>(50L*x0)];
                        auto tmp4 = out_ptr1[static_cast<long>(50L*x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1, 64), (64, 64, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (1, 1, 128), (128, 128, 1))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (1, 1, 320), (320, 320, 1))
    assert_size_stride(arg19_1, (320, ), (1, ))
    assert_size_stride(arg20_1, (320, ), (1, ))
    assert_size_stride(arg21_1, (320, ), (1, ))
    assert_size_stride(arg22_1, (320, ), (1, ))
    assert_size_stride(arg23_1, (320, ), (1, ))
    assert_size_stride(arg24_1, (320, ), (1, ))
    assert_size_stride(arg25_1, (320, ), (1, ))
    assert_size_stride(arg26_1, (320, ), (1, ))
    assert_size_stride(arg27_1, (1, 1, 512), (512, 512, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, ), (1, ))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, ), (1, ))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (64, 3, 4, 4), (48, 16, 4, 1))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (64, ), (1, ))
    assert_size_stride(arg41_1, (64, ), (1, ))
    assert_size_stride(arg42_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (192, 64), (64, 1))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg47_1, (16, ), (1, ))
    assert_size_stride(arg48_1, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg49_1, (24, ), (1, ))
    assert_size_stride(arg50_1, (24, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg51_1, (24, ), (1, ))
    assert_size_stride(arg52_1, (64, 64), (64, 1))
    assert_size_stride(arg53_1, (64, ), (1, ))
    assert_size_stride(arg54_1, (512, 64), (64, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (64, 512), (512, 1))
    assert_size_stride(arg57_1, (64, ), (1, ))
    assert_size_stride(arg58_1, (192, 64), (64, 1))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (64, 64), (64, 1))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (512, 64), (64, 1))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (64, 512), (512, 1))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (128, 64, 2, 2), (256, 4, 2, 1))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (384, 128), (128, 1))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg75_1, (32, ), (1, ))
    assert_size_stride(arg76_1, (48, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg77_1, (48, ), (1, ))
    assert_size_stride(arg78_1, (48, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg79_1, (48, ), (1, ))
    assert_size_stride(arg80_1, (128, 128), (128, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (1024, 128), (128, 1))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (128, 1024), (1024, 1))
    assert_size_stride(arg85_1, (128, ), (1, ))
    assert_size_stride(arg86_1, (384, 128), (128, 1))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (128, 128), (128, 1))
    assert_size_stride(arg89_1, (128, ), (1, ))
    assert_size_stride(arg90_1, (1024, 128), (128, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (128, 1024), (1024, 1))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (320, 128, 2, 2), (512, 4, 2, 1))
    assert_size_stride(arg95_1, (320, ), (1, ))
    assert_size_stride(arg96_1, (320, ), (1, ))
    assert_size_stride(arg97_1, (320, ), (1, ))
    assert_size_stride(arg98_1, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg99_1, (320, ), (1, ))
    assert_size_stride(arg100_1, (960, 320), (320, 1))
    assert_size_stride(arg101_1, (960, ), (1, ))
    assert_size_stride(arg102_1, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg103_1, (80, ), (1, ))
    assert_size_stride(arg104_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg105_1, (120, ), (1, ))
    assert_size_stride(arg106_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg107_1, (120, ), (1, ))
    assert_size_stride(arg108_1, (320, 320), (320, 1))
    assert_size_stride(arg109_1, (320, ), (1, ))
    assert_size_stride(arg110_1, (1280, 320), (320, 1))
    assert_size_stride(arg111_1, (1280, ), (1, ))
    assert_size_stride(arg112_1, (320, 1280), (1280, 1))
    assert_size_stride(arg113_1, (320, ), (1, ))
    assert_size_stride(arg114_1, (960, 320), (320, 1))
    assert_size_stride(arg115_1, (960, ), (1, ))
    assert_size_stride(arg116_1, (320, 320), (320, 1))
    assert_size_stride(arg117_1, (320, ), (1, ))
    assert_size_stride(arg118_1, (1280, 320), (320, 1))
    assert_size_stride(arg119_1, (1280, ), (1, ))
    assert_size_stride(arg120_1, (320, 1280), (1280, 1))
    assert_size_stride(arg121_1, (320, ), (1, ))
    assert_size_stride(arg122_1, (512, 320, 2, 2), (1280, 4, 2, 1))
    assert_size_stride(arg123_1, (512, ), (1, ))
    assert_size_stride(arg124_1, (512, ), (1, ))
    assert_size_stride(arg125_1, (512, ), (1, ))
    assert_size_stride(arg126_1, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (1536, 512), (512, 1))
    assert_size_stride(arg129_1, (1536, ), (1, ))
    assert_size_stride(arg130_1, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (192, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg135_1, (192, ), (1, ))
    assert_size_stride(arg136_1, (512, 512), (512, 1))
    assert_size_stride(arg137_1, (512, ), (1, ))
    assert_size_stride(arg138_1, (2048, 512), (512, 1))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (512, 2048), (2048, 1))
    assert_size_stride(arg141_1, (512, ), (1, ))
    assert_size_stride(arg142_1, (1536, 512), (512, 1))
    assert_size_stride(arg143_1, (1536, ), (1, ))
    assert_size_stride(arg144_1, (512, 512), (512, 1))
    assert_size_stride(arg145_1, (512, ), (1, ))
    assert_size_stride(arg146_1, (2048, 512), (512, 1))
    assert_size_stride(arg147_1, (2048, ), (1, ))
    assert_size_stride(arg148_1, (512, 2048), (2048, 1))
    assert_size_stride(arg149_1, (512, ), (1, ))
    assert_size_stride(arg150_1, (1000, 512), (512, 1))
    assert_size_stride(arg151_1, (1000, ), (1, ))
    assert_size_stride(arg152_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 4, 4), (48, 1, 12, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg152_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg152_1
    del arg38_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg39_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg39_1
    del buf0
    del buf1
    buf3 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 3136, 1), (3136, 1, 25088), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 3137, 64), (200768, 1, 3137), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg0_1
    del arg40_1
    del arg41_1
    del buf2
    del buf3
    del buf4
    # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, arg42_1, arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64)
    assert_size_stride(buf8, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del buf7
    buf9 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((8, 3137, 1), (3137, 1, 25096), device='cpu', dtype=torch.float32)
    buf12 = empty((8, 3137, 64), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_2(c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg1_1
    del arg2_1
    buf13 = empty((25096, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg45_1, reinterpret_tensor(buf12, (25096, 64), (64, 1), 0), reinterpret_tensor(arg44_1, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf13)
    del arg44_1
    del arg45_1
    buf14 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cpu', dtype=torch.float32)
    buf15 = reinterpret_tensor(buf12, (8, 8, 3137, 8), (200768, 8, 64, 1), 0); del buf12  # reuse
    buf16 = empty_strided((8, 8, 1, 8), (64, 8, 512, 1), device='cpu', dtype=torch.float32)
    buf17 = empty((8, 8, 3137, 8), device='cpu', dtype=torch.float32)
    buf18 = empty((8, 8, 3137, 8), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_3(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    buf19 = empty((64, 8, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf17, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf18, (64, 3137, 8), (25096, 8, 1), 0), out=buf19)
    buf20 = buf18; del buf18  # reuse
    cpp_fused_clone_4(c_void_p(buf13.data_ptr()), c_void_p(buf20.data_ptr()))
    buf21 = reinterpret_tensor(buf17, (64, 3137, 8), (25096, 8, 1), 0); del buf17  # reuse
    # Source Nodes: [factor_att_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf20, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf19, (64, 8, 8), (64, 8, 1), 0), out=buf21)
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(reinterpret_tensor(buf13, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), arg46_1, arg47_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16)
    assert_size_stride(buf22, (8, 16, 56, 56), (50176, 1, 896, 16))
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf23 = extern_kernels.convolution(reinterpret_tensor(buf13, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), arg48_1, arg49_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf23, (8, 24, 56, 56), (75264, 1, 1344, 24))
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(reinterpret_tensor(buf13, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), arg50_1, arg51_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf24, (8, 24, 56, 56), (75264, 1, 1344, 24))
    buf25 = reinterpret_tensor(buf20, (8, 3137, 8, 8), (200768, 64, 8, 1), 0); del buf20  # reuse
    cpp_fused_clone_5(c_void_p(buf21.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del buf22
    del buf23
    del buf24
    buf26 = reinterpret_tensor(buf21, (25096, 64), (64, 1), 0); del buf21  # reuse
    # Source Nodes: [x_11], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf25, (25096, 64), (64, 1), 0), reinterpret_tensor(arg52_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf26)
    del arg52_1
    del arg53_1
    buf27 = buf9; del buf9  # reuse
    buf28 = buf10; del buf10  # reuse
    buf30 = reinterpret_tensor(buf25, (8, 3137, 64), (200768, 64, 1), 0); del buf25  # reuse
    cpp_fused_add_cat_native_layer_norm_6(c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()))
    del arg3_1
    del arg4_1
    buf31 = empty((25096, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_15], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf30, (25096, 64), (64, 1), 0), reinterpret_tensor(arg54_1, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf31)
    del arg54_1
    del arg55_1
    buf32 = reinterpret_tensor(buf31, (8, 3137, 512), (1606144, 512, 1), 0); del buf31  # reuse
    cpp_fused_gelu_7(c_void_p(buf32.data_ptr()))
    buf33 = reinterpret_tensor(buf30, (25096, 64), (64, 1), 0); del buf30  # reuse
    # Source Nodes: [x_19], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg57_1, reinterpret_tensor(buf32, (25096, 512), (512, 1), 0), reinterpret_tensor(arg56_1, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf33)
    del arg56_1
    del arg57_1
    buf34 = reinterpret_tensor(buf33, (8, 3137, 64), (200768, 64, 1), 0); del buf33  # reuse
    cpp_fused_add_cat_8(c_void_p(buf34.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf26.data_ptr()))
    # Source Nodes: [l__mod___serial_blocks1_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf35 = extern_kernels.convolution(reinterpret_tensor(buf34, (8, 64, 56, 56), (200768, 1, 3584, 64), 64), arg42_1, arg43_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64)
    assert_size_stride(buf35, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg42_1
    del arg43_1
    buf36 = buf28; del buf28  # reuse
    buf37 = buf27; del buf27  # reuse
    buf39 = reinterpret_tensor(buf6, (8, 3137, 64), (200768, 64, 1), 0); del buf6  # reuse
    cpp_fused_cat_native_layer_norm_9(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf39.data_ptr()))
    del arg5_1
    del arg6_1
    buf40 = buf13; del buf13  # reuse
    # Source Nodes: [l__mod___serial_blocks1_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf39, (25096, 64), (64, 1), 0), reinterpret_tensor(arg58_1, (64, 192), (1, 64), 0), alpha=1, beta=1, out=buf40)
    del arg58_1
    del arg59_1
    buf41 = buf16; del buf16  # reuse
    buf42 = reinterpret_tensor(buf39, (8, 8, 3137, 8), (200768, 8, 64, 1), 0); del buf39  # reuse
    buf43 = buf14; del buf14  # reuse
    buf44 = reinterpret_tensor(buf26, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf26  # reuse
    buf45 = reinterpret_tensor(buf15, (8, 8, 3137, 8), (200768, 25096, 8, 1), 0); del buf15  # reuse
    cpp_fused__softmax_clone_10(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()))
    del buf41
    del buf42
    del buf43
    buf46 = buf19; del buf19  # reuse
    # Source Nodes: [factor_att_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf44, (64, 8, 3137), (25096, 1, 8), 0), reinterpret_tensor(buf45, (64, 3137, 8), (25096, 8, 1), 0), out=buf46)
    buf47 = buf45; del buf45  # reuse
    cpp_fused_clone_11(c_void_p(buf40.data_ptr()), c_void_p(buf47.data_ptr()))
    buf48 = reinterpret_tensor(buf44, (64, 3137, 8), (25096, 8, 1), 0); del buf44  # reuse
    # Source Nodes: [factor_att_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf47, (64, 3137, 8), (25096, 8, 1), 0), reinterpret_tensor(buf46, (64, 8, 8), (64, 8, 1), 0), out=buf48)
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(reinterpret_tensor(buf40, (8, 16, 56, 56), (602304, 1, 10752, 192), 320), arg46_1, arg47_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16)
    assert_size_stride(buf49, (8, 16, 56, 56), (50176, 1, 896, 16))
    del arg46_1
    del arg47_1
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(reinterpret_tensor(buf40, (8, 24, 56, 56), (602304, 1, 10752, 192), 336), arg48_1, arg49_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf50, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg48_1
    del arg49_1
    # Source Nodes: [l__mod___serial_blocks1_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(reinterpret_tensor(buf40, (8, 24, 56, 56), (602304, 1, 10752, 192), 360), arg50_1, arg51_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24)
    assert_size_stride(buf51, (8, 24, 56, 56), (75264, 1, 1344, 24))
    del arg50_1
    del arg51_1
    buf52 = reinterpret_tensor(buf47, (8, 3137, 8, 8), (200768, 64, 8, 1), 0); del buf47  # reuse
    cpp_fused_clone_12(c_void_p(buf48.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del buf40
    del buf49
    del buf50
    del buf51
    buf53 = reinterpret_tensor(buf48, (25096, 64), (64, 1), 0); del buf48  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg61_1, reinterpret_tensor(buf52, (25096, 64), (64, 1), 0), reinterpret_tensor(arg60_1, (64, 64), (1, 64), 0), alpha=1, beta=1, out=buf53)
    del arg60_1
    del arg61_1
    buf54 = buf37; del buf37  # reuse
    buf55 = buf36; del buf36  # reuse
    buf57 = reinterpret_tensor(buf52, (8, 3137, 64), (200768, 64, 1), 0); del buf52  # reuse
    cpp_fused_add_cat_native_layer_norm_13(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg7_1
    del arg8_1
    del buf54
    del buf55
    buf58 = reinterpret_tensor(buf32, (25096, 512), (512, 1), 0); del buf32  # reuse
    # Source Nodes: [x_33], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg63_1, reinterpret_tensor(buf57, (25096, 64), (64, 1), 0), reinterpret_tensor(arg62_1, (64, 512), (1, 64), 0), alpha=1, beta=1, out=buf58)
    del arg62_1
    del arg63_1
    buf59 = reinterpret_tensor(buf58, (8, 3137, 512), (1606144, 512, 1), 0); del buf58  # reuse
    cpp_fused_gelu_14(c_void_p(buf59.data_ptr()))
    buf60 = reinterpret_tensor(buf57, (25096, 64), (64, 1), 0); del buf57  # reuse
    # Source Nodes: [x_37], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg65_1, reinterpret_tensor(buf59, (25096, 512), (512, 1), 0), reinterpret_tensor(arg64_1, (512, 64), (1, 512), 0), alpha=1, beta=1, out=buf60)
    del arg64_1
    del arg65_1
    del buf59
    buf61 = buf8; del buf8  # reuse
    buf62 = empty_strided((128, 64, 2, 2), (256, 1, 128, 64), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_15(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del arg66_1
    del buf34
    del buf35
    del buf53
    del buf60
    # Source Nodes: [x1_nocls, x_40], Original ATen: [aten.clone, aten.convolution]
    buf63 = extern_kernels.convolution(buf61, buf62, arg67_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf63, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del arg67_1
    del buf61
    del buf62
    buf64 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf65 = empty_strided((8, 784, 1), (784, 1, 6272), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((8, 785, 128), (100480, 1, 785), device='cpu', dtype=torch.float32)
    buf68 = empty_strided((8, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_native_layer_norm_16(c_void_p(buf63.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()))
    del arg68_1
    del arg69_1
    del arg9_1
    del buf63
    del buf64
    del buf65
    # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj], Original ATen: [aten.convolution]
    buf69 = extern_kernels.convolution(buf68, arg70_1, arg71_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf69, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del buf68
    buf70 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf71 = empty_strided((8, 785, 1), (785, 1, 6280), device='cpu', dtype=torch.float32)
    buf73 = empty((8, 785, 128), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_17(c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf73.data_ptr()))
    del arg10_1
    del arg11_1
    buf74 = empty((6280, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf73, (6280, 128), (128, 1), 0), reinterpret_tensor(arg72_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf74)
    del arg72_1
    del arg73_1
    buf75 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cpu', dtype=torch.float32)
    buf76 = reinterpret_tensor(buf73, (8, 8, 785, 16), (100480, 16, 128, 1), 0); del buf73  # reuse
    buf77 = empty_strided((8, 8, 1, 16), (128, 16, 1024, 1), device='cpu', dtype=torch.float32)
    buf78 = empty((8, 8, 785, 16), device='cpu', dtype=torch.float32)
    buf79 = empty((8, 8, 785, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_18(c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    buf80 = empty((64, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf78, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf79, (64, 785, 16), (12560, 16, 1), 0), out=buf80)
    buf81 = buf79; del buf79  # reuse
    cpp_fused_clone_19(c_void_p(buf74.data_ptr()), c_void_p(buf81.data_ptr()))
    buf82 = reinterpret_tensor(buf78, (64, 785, 16), (12560, 16, 1), 0); del buf78  # reuse
    # Source Nodes: [factor_att_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf80, (64, 16, 16), (256, 16, 1), 0), out=buf82)
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf83 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), arg74_1, arg75_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32)
    assert_size_stride(buf83, (8, 32, 28, 28), (25088, 1, 896, 32))
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), arg76_1, arg77_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf84, (8, 48, 28, 28), (37632, 1, 1344, 48))
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf85 = extern_kernels.convolution(reinterpret_tensor(buf74, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), arg78_1, arg79_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf85, (8, 48, 28, 28), (37632, 1, 1344, 48))
    buf86 = reinterpret_tensor(buf81, (8, 785, 8, 16), (100480, 128, 16, 1), 0); del buf81  # reuse
    cpp_fused_clone_20(c_void_p(buf82.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    del buf83
    del buf84
    del buf85
    buf87 = reinterpret_tensor(buf82, (6280, 128), (128, 1), 0); del buf82  # reuse
    # Source Nodes: [x_51], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf86, (6280, 128), (128, 1), 0), reinterpret_tensor(arg80_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf87)
    del arg80_1
    del arg81_1
    buf88 = buf71; del buf71  # reuse
    buf89 = buf70; del buf70  # reuse
    buf91 = reinterpret_tensor(buf86, (8, 785, 128), (100480, 128, 1), 0); del buf86  # reuse
    cpp_fused_add_cat_native_layer_norm_21(c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()))
    del arg12_1
    del arg13_1
    buf92 = empty((6280, 1024), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_55], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, reinterpret_tensor(buf91, (6280, 128), (128, 1), 0), reinterpret_tensor(arg82_1, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf92)
    del arg82_1
    del arg83_1
    buf93 = reinterpret_tensor(buf92, (8, 785, 1024), (803840, 1024, 1), 0); del buf92  # reuse
    cpp_fused_gelu_22(c_void_p(buf93.data_ptr()))
    buf94 = reinterpret_tensor(buf91, (6280, 128), (128, 1), 0); del buf91  # reuse
    # Source Nodes: [x_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf93, (6280, 1024), (1024, 1), 0), reinterpret_tensor(arg84_1, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf94)
    del arg84_1
    del arg85_1
    buf95 = reinterpret_tensor(buf94, (8, 785, 128), (100480, 128, 1), 0); del buf94  # reuse
    cpp_fused_add_cat_23(c_void_p(buf95.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf87.data_ptr()))
    # Source Nodes: [l__mod___serial_blocks2_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(reinterpret_tensor(buf95, (8, 128, 28, 28), (100480, 1, 3584, 128), 128), arg70_1, arg71_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf96, (8, 128, 28, 28), (100352, 1, 3584, 128))
    del arg70_1
    del arg71_1
    buf97 = buf89; del buf89  # reuse
    buf98 = buf88; del buf88  # reuse
    buf100 = reinterpret_tensor(buf87, (8, 785, 128), (100480, 128, 1), 0); del buf87  # reuse
    cpp_fused_cat_native_layer_norm_24(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf100.data_ptr()))
    del arg14_1
    del arg15_1
    buf101 = buf74; del buf74  # reuse
    # Source Nodes: [l__mod___serial_blocks2_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf100, (6280, 128), (128, 1), 0), reinterpret_tensor(arg86_1, (128, 384), (1, 128), 0), alpha=1, beta=1, out=buf101)
    del arg86_1
    del arg87_1
    buf102 = buf77; del buf77  # reuse
    buf103 = reinterpret_tensor(buf100, (8, 8, 785, 16), (100480, 16, 128, 1), 0); del buf100  # reuse
    buf104 = buf75; del buf75  # reuse
    buf105 = reinterpret_tensor(buf67, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf67  # reuse
    buf106 = reinterpret_tensor(buf76, (8, 8, 785, 16), (100480, 12560, 16, 1), 0); del buf76  # reuse
    cpp_fused__softmax_clone_25(c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del buf102
    del buf103
    del buf104
    buf107 = buf80; del buf80  # reuse
    # Source Nodes: [factor_att_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf105, (64, 16, 785), (12560, 1, 16), 0), reinterpret_tensor(buf106, (64, 785, 16), (12560, 16, 1), 0), out=buf107)
    buf108 = buf106; del buf106  # reuse
    cpp_fused_clone_26(c_void_p(buf101.data_ptr()), c_void_p(buf108.data_ptr()))
    buf109 = reinterpret_tensor(buf105, (64, 785, 16), (12560, 16, 1), 0); del buf105  # reuse
    # Source Nodes: [factor_att_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf108, (64, 785, 16), (12560, 16, 1), 0), reinterpret_tensor(buf107, (64, 16, 16), (256, 16, 1), 0), out=buf109)
    del buf107
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf110 = extern_kernels.convolution(reinterpret_tensor(buf101, (8, 32, 28, 28), (301440, 1, 10752, 384), 640), arg74_1, arg75_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32)
    assert_size_stride(buf110, (8, 32, 28, 28), (25088, 1, 896, 32))
    del arg74_1
    del arg75_1
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf111 = extern_kernels.convolution(reinterpret_tensor(buf101, (8, 48, 28, 28), (301440, 1, 10752, 384), 672), arg76_1, arg77_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf111, (8, 48, 28, 28), (37632, 1, 1344, 48))
    del arg76_1
    del arg77_1
    # Source Nodes: [l__mod___serial_blocks2_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf112 = extern_kernels.convolution(reinterpret_tensor(buf101, (8, 48, 28, 28), (301440, 1, 10752, 384), 720), arg78_1, arg79_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48)
    assert_size_stride(buf112, (8, 48, 28, 28), (37632, 1, 1344, 48))
    del arg78_1
    del arg79_1
    buf113 = reinterpret_tensor(buf108, (8, 785, 8, 16), (100480, 128, 16, 1), 0); del buf108  # reuse
    cpp_fused_clone_27(c_void_p(buf109.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    del buf101
    del buf111
    del buf112
    buf114 = reinterpret_tensor(buf109, (6280, 128), (128, 1), 0); del buf109  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf113, (6280, 128), (128, 1), 0), reinterpret_tensor(arg88_1, (128, 128), (1, 128), 0), alpha=1, beta=1, out=buf114)
    del arg88_1
    del arg89_1
    buf115 = buf98; del buf98  # reuse
    buf116 = buf97; del buf97  # reuse
    buf118 = reinterpret_tensor(buf113, (8, 785, 128), (100480, 128, 1), 0); del buf113  # reuse
    cpp_fused_add_cat_native_layer_norm_28(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg16_1
    del arg17_1
    del buf115
    del buf116
    buf119 = reinterpret_tensor(buf93, (6280, 1024), (1024, 1), 0); del buf93  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf118, (6280, 128), (128, 1), 0), reinterpret_tensor(arg90_1, (128, 1024), (1, 128), 0), alpha=1, beta=1, out=buf119)
    del arg90_1
    del arg91_1
    buf120 = reinterpret_tensor(buf119, (8, 785, 1024), (803840, 1024, 1), 0); del buf119  # reuse
    cpp_fused_gelu_29(c_void_p(buf120.data_ptr()))
    buf121 = reinterpret_tensor(buf118, (6280, 128), (128, 1), 0); del buf118  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf120, (6280, 1024), (1024, 1), 0), reinterpret_tensor(arg92_1, (1024, 128), (1, 1024), 0), alpha=1, beta=1, out=buf121)
    del arg92_1
    del arg93_1
    del buf120
    buf122 = buf69; del buf69  # reuse
    buf123 = empty_strided((320, 128, 2, 2), (512, 1, 256, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_30(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del arg94_1
    del buf114
    del buf121
    del buf95
    del buf96
    # Source Nodes: [x2_nocls, x_80], Original ATen: [aten.clone, aten.convolution]
    buf124 = extern_kernels.convolution(buf122, buf123, arg95_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf124, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del arg95_1
    del buf122
    del buf123
    buf125 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf126 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf128 = empty_strided((8, 197, 320), (63040, 1, 197), device='cpu', dtype=torch.float32)
    buf129 = empty_strided((8, 320, 14, 14), (62720, 1, 4480, 320), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_native_layer_norm_31(c_void_p(buf124.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg18_1
    del arg96_1
    del arg97_1
    del buf124
    del buf125
    del buf126
    # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj], Original ATen: [aten.convolution]
    buf130 = extern_kernels.convolution(buf129, arg98_1, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320)
    assert_size_stride(buf130, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del buf129
    buf131 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf132 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf134 = empty((8, 197, 320), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_32(c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()))
    del arg19_1
    del arg20_1
    buf135 = empty((1576, 960), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf134, (1576, 320), (320, 1), 0), reinterpret_tensor(arg100_1, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf135)
    del arg100_1
    del arg101_1
    buf136 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cpu', dtype=torch.float32)
    buf137 = reinterpret_tensor(buf134, (8, 8, 197, 40), (63040, 40, 320, 1), 0); del buf134  # reuse
    buf138 = empty_strided((8, 8, 1, 40), (320, 40, 2560, 1), device='cpu', dtype=torch.float32)
    buf139 = empty((8, 8, 197, 40), device='cpu', dtype=torch.float32)
    buf140 = empty((8, 8, 197, 40), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_33(c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()))
    buf141 = empty((64, 40, 40), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf139, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf140, (64, 197, 40), (7880, 40, 1), 0), out=buf141)
    buf142 = buf140; del buf140  # reuse
    cpp_fused_clone_34(c_void_p(buf135.data_ptr()), c_void_p(buf142.data_ptr()))
    buf143 = reinterpret_tensor(buf139, (64, 197, 40), (7880, 40, 1), 0); del buf139  # reuse
    # Source Nodes: [factor_att_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf142, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf141, (64, 40, 40), (1600, 40, 1), 0), out=buf143)
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf144 = extern_kernels.convolution(reinterpret_tensor(buf135, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), arg102_1, arg103_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80)
    assert_size_stride(buf144, (8, 80, 14, 14), (15680, 1, 1120, 80))
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf145 = extern_kernels.convolution(reinterpret_tensor(buf135, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), arg104_1, arg105_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf145, (8, 120, 14, 14), (23520, 1, 1680, 120))
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf146 = extern_kernels.convolution(reinterpret_tensor(buf135, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), arg106_1, arg107_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf146, (8, 120, 14, 14), (23520, 1, 1680, 120))
    buf147 = reinterpret_tensor(buf142, (8, 197, 8, 40), (63040, 320, 40, 1), 0); del buf142  # reuse
    cpp_fused_clone_35(c_void_p(buf143.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del buf144
    del buf145
    del buf146
    buf148 = reinterpret_tensor(buf143, (1576, 320), (320, 1), 0); del buf143  # reuse
    # Source Nodes: [x_91], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf147, (1576, 320), (320, 1), 0), reinterpret_tensor(arg108_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf148)
    del arg108_1
    del arg109_1
    buf149 = buf132; del buf132  # reuse
    buf150 = buf131; del buf131  # reuse
    buf152 = reinterpret_tensor(buf147, (8, 197, 320), (63040, 320, 1), 0); del buf147  # reuse
    cpp_fused_add_cat_native_layer_norm_36(c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf152.data_ptr()))
    del arg21_1
    del arg22_1
    buf153 = empty((1576, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf152, (1576, 320), (320, 1), 0), reinterpret_tensor(arg110_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf153)
    del arg110_1
    del arg111_1
    buf154 = reinterpret_tensor(buf153, (8, 197, 1280), (252160, 1280, 1), 0); del buf153  # reuse
    cpp_fused_gelu_37(c_void_p(buf154.data_ptr()))
    buf155 = reinterpret_tensor(buf152, (1576, 320), (320, 1), 0); del buf152  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf154, (1576, 1280), (1280, 1), 0), reinterpret_tensor(arg112_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf155)
    del arg112_1
    del arg113_1
    buf156 = reinterpret_tensor(buf155, (8, 197, 320), (63040, 320, 1), 0); del buf155  # reuse
    cpp_fused_add_cat_38(c_void_p(buf156.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf148.data_ptr()))
    # Source Nodes: [l__mod___serial_blocks3_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf157 = extern_kernels.convolution(reinterpret_tensor(buf156, (8, 320, 14, 14), (63040, 1, 4480, 320), 320), arg98_1, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320)
    assert_size_stride(buf157, (8, 320, 14, 14), (62720, 1, 4480, 320))
    del arg98_1
    del arg99_1
    buf158 = buf150; del buf150  # reuse
    buf159 = buf149; del buf149  # reuse
    buf161 = reinterpret_tensor(buf148, (8, 197, 320), (63040, 320, 1), 0); del buf148  # reuse
    cpp_fused_cat_native_layer_norm_39(c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    del arg23_1
    del arg24_1
    buf162 = buf135; del buf135  # reuse
    # Source Nodes: [l__mod___serial_blocks3_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf161, (1576, 320), (320, 1), 0), reinterpret_tensor(arg114_1, (320, 960), (1, 320), 0), alpha=1, beta=1, out=buf162)
    del arg114_1
    del arg115_1
    buf163 = buf138; del buf138  # reuse
    buf164 = reinterpret_tensor(buf161, (8, 8, 197, 40), (63040, 40, 320, 1), 0); del buf161  # reuse
    buf165 = buf136; del buf136  # reuse
    buf166 = reinterpret_tensor(buf128, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf128  # reuse
    buf167 = reinterpret_tensor(buf137, (8, 8, 197, 40), (63040, 7880, 40, 1), 0); del buf137  # reuse
    cpp_fused__softmax_clone_40(c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    del buf163
    del buf164
    del buf165
    buf168 = buf141; del buf141  # reuse
    # Source Nodes: [factor_att_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf166, (64, 40, 197), (7880, 1, 40), 0), reinterpret_tensor(buf167, (64, 197, 40), (7880, 40, 1), 0), out=buf168)
    buf169 = buf167; del buf167  # reuse
    cpp_fused_clone_41(c_void_p(buf162.data_ptr()), c_void_p(buf169.data_ptr()))
    buf170 = reinterpret_tensor(buf166, (64, 197, 40), (7880, 40, 1), 0); del buf166  # reuse
    # Source Nodes: [factor_att_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (64, 197, 40), (7880, 40, 1), 0), reinterpret_tensor(buf168, (64, 40, 40), (1600, 40, 1), 0), out=buf170)
    del buf168
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf171 = extern_kernels.convolution(reinterpret_tensor(buf162, (8, 80, 14, 14), (189120, 1, 13440, 960), 1600), arg102_1, arg103_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80)
    assert_size_stride(buf171, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg102_1
    del arg103_1
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf172 = extern_kernels.convolution(reinterpret_tensor(buf162, (8, 120, 14, 14), (189120, 1, 13440, 960), 1680), arg104_1, arg105_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf172, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg104_1
    del arg105_1
    # Source Nodes: [l__mod___serial_blocks3_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf173 = extern_kernels.convolution(reinterpret_tensor(buf162, (8, 120, 14, 14), (189120, 1, 13440, 960), 1800), arg106_1, arg107_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120)
    assert_size_stride(buf173, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg106_1
    del arg107_1
    buf174 = reinterpret_tensor(buf169, (8, 197, 8, 40), (63040, 320, 40, 1), 0); del buf169  # reuse
    cpp_fused_clone_42(c_void_p(buf170.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del buf162
    del buf171
    del buf172
    del buf173
    buf175 = reinterpret_tensor(buf170, (1576, 320), (320, 1), 0); del buf170  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf174, (1576, 320), (320, 1), 0), reinterpret_tensor(arg116_1, (320, 320), (1, 320), 0), alpha=1, beta=1, out=buf175)
    del arg116_1
    del arg117_1
    buf176 = buf159; del buf159  # reuse
    buf177 = buf158; del buf158  # reuse
    buf179 = reinterpret_tensor(buf174, (8, 197, 320), (63040, 320, 1), 0); del buf174  # reuse
    cpp_fused_add_cat_native_layer_norm_43(c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()))
    del arg25_1
    del arg26_1
    del buf176
    del buf177
    buf180 = reinterpret_tensor(buf154, (1576, 1280), (1280, 1), 0); del buf154  # reuse
    # Source Nodes: [x_113], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf179, (1576, 320), (320, 1), 0), reinterpret_tensor(arg118_1, (320, 1280), (1, 320), 0), alpha=1, beta=1, out=buf180)
    del arg118_1
    del arg119_1
    buf181 = reinterpret_tensor(buf180, (8, 197, 1280), (252160, 1280, 1), 0); del buf180  # reuse
    cpp_fused_gelu_44(c_void_p(buf181.data_ptr()))
    buf182 = reinterpret_tensor(buf179, (1576, 320), (320, 1), 0); del buf179  # reuse
    # Source Nodes: [x_117], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf181, (1576, 1280), (1280, 1), 0), reinterpret_tensor(arg120_1, (1280, 320), (1, 1280), 0), alpha=1, beta=1, out=buf182)
    del arg120_1
    del arg121_1
    del buf181
    buf183 = buf130; del buf130  # reuse
    buf184 = empty_strided((512, 320, 2, 2), (1280, 1, 640, 320), device='cpu', dtype=torch.float32)
    cpp_fused_clone_convolution_45(c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg122_1
    del buf156
    del buf157
    del buf175
    del buf182
    # Source Nodes: [x3_nocls, x_120], Original ATen: [aten.clone, aten.convolution]
    buf185 = extern_kernels.convolution(buf183, buf184, arg123_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf185, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del arg123_1
    del buf183
    del buf184
    buf186 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf187 = empty_strided((8, 49, 1), (49, 1, 392), device='cpu', dtype=torch.float32)
    buf189 = empty_strided((8, 50, 512), (25600, 1, 50), device='cpu', dtype=torch.float32)
    buf190 = reinterpret_tensor(buf110, (8, 512, 7, 7), (25088, 1, 3584, 512), 0); del buf110  # reuse
    cpp_fused_cat_convolution_native_layer_norm_46(c_void_p(buf185.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    del arg124_1
    del arg125_1
    del arg27_1
    del buf185
    del buf186
    del buf187
    # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj], Original ATen: [aten.convolution]
    buf191 = extern_kernels.convolution(buf190, arg126_1, arg127_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf191, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del buf190
    buf192 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf193 = empty_strided((8, 50, 1), (50, 1, 400), device='cpu', dtype=torch.float32)
    buf195 = empty((8, 50, 512), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_47(c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf195.data_ptr()))
    del arg28_1
    del arg29_1
    buf196 = empty((400, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf195, (400, 512), (512, 1), 0), reinterpret_tensor(arg128_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf196)
    del arg128_1
    del arg129_1
    buf197 = reinterpret_tensor(buf46, (8, 8, 1, 64), (512, 64, 4096, 1), 0); del buf46  # reuse
    buf198 = reinterpret_tensor(buf195, (8, 8, 50, 64), (25600, 64, 512, 1), 0); del buf195  # reuse
    buf199 = empty_strided((8, 8, 1, 64), (512, 64, 4096, 1), device='cpu', dtype=torch.float32)
    buf200 = empty((8, 8, 50, 64), device='cpu', dtype=torch.float32)
    buf201 = empty((8, 8, 50, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_48(c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    buf202 = empty((64, 64, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [factor_att_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf200, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf201, (64, 50, 64), (3200, 64, 1), 0), out=buf202)
    buf203 = buf201; del buf201  # reuse
    cpp_fused_clone_49(c_void_p(buf196.data_ptr()), c_void_p(buf203.data_ptr()))
    buf204 = reinterpret_tensor(buf200, (64, 50, 64), (3200, 64, 1), 0); del buf200  # reuse
    # Source Nodes: [factor_att_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf203, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf202, (64, 64, 64), (4096, 64, 1), 0), out=buf204)
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_0], Original ATen: [aten.convolution]
    buf205 = extern_kernels.convolution(reinterpret_tensor(buf196, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), arg130_1, arg131_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf205, (8, 128, 7, 7), (6272, 1, 896, 128))
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_1], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(reinterpret_tensor(buf196, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), arg132_1, arg133_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf206, (8, 192, 7, 7), (9408, 1, 1344, 192))
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_2], Original ATen: [aten.convolution]
    buf207 = extern_kernels.convolution(reinterpret_tensor(buf196, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), arg134_1, arg135_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf207, (8, 192, 7, 7), (9408, 1, 1344, 192))
    buf208 = reinterpret_tensor(buf203, (8, 50, 8, 64), (25600, 512, 64, 1), 0); del buf203  # reuse
    cpp_fused_clone_50(c_void_p(buf204.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del buf205
    del buf206
    del buf207
    buf209 = reinterpret_tensor(buf204, (400, 512), (512, 1), 0); del buf204  # reuse
    # Source Nodes: [x_131], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf208, (400, 512), (512, 1), 0), reinterpret_tensor(arg136_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf209)
    del arg136_1
    del arg137_1
    buf210 = buf193; del buf193  # reuse
    buf211 = buf192; del buf192  # reuse
    buf213 = reinterpret_tensor(buf208, (8, 50, 512), (25600, 512, 1), 0); del buf208  # reuse
    cpp_fused_add_cat_native_layer_norm_51(c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()))
    del arg30_1
    del arg31_1
    buf214 = empty((400, 2048), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_135], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf213, (400, 512), (512, 1), 0), reinterpret_tensor(arg138_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf214)
    del arg138_1
    del arg139_1
    buf215 = reinterpret_tensor(buf214, (8, 50, 2048), (102400, 2048, 1), 0); del buf214  # reuse
    cpp_fused_gelu_52(c_void_p(buf215.data_ptr()))
    buf216 = reinterpret_tensor(buf213, (400, 512), (512, 1), 0); del buf213  # reuse
    # Source Nodes: [x_139], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf215, (400, 2048), (2048, 1), 0), reinterpret_tensor(arg140_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf216)
    del arg140_1
    del arg141_1
    buf217 = reinterpret_tensor(buf216, (8, 50, 512), (25600, 512, 1), 0); del buf216  # reuse
    cpp_fused_add_cat_53(c_void_p(buf217.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf209.data_ptr()))
    del buf191
    # Source Nodes: [l__mod___serial_blocks4_0_cpe_proj_1], Original ATen: [aten.convolution]
    buf218 = extern_kernels.convolution(reinterpret_tensor(buf217, (8, 512, 7, 7), (25600, 1, 3584, 512), 512), arg126_1, arg127_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512)
    assert_size_stride(buf218, (8, 512, 7, 7), (25088, 1, 3584, 512))
    del arg126_1
    del arg127_1
    buf219 = buf211; del buf211  # reuse
    buf220 = buf210; del buf210  # reuse
    buf222 = reinterpret_tensor(buf209, (8, 50, 512), (25600, 512, 1), 0); del buf209  # reuse
    cpp_fused_cat_native_layer_norm_54(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()))
    del arg32_1
    del arg33_1
    buf223 = buf196; del buf196  # reuse
    # Source Nodes: [l__mod___serial_blocks4_1_factoratt_crpe_qkv], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf222, (400, 512), (512, 1), 0), reinterpret_tensor(arg142_1, (512, 1536), (1, 512), 0), alpha=1, beta=1, out=buf223)
    del arg142_1
    del arg143_1
    buf224 = buf199; del buf199  # reuse
    buf225 = reinterpret_tensor(buf222, (8, 8, 50, 64), (25600, 64, 512, 1), 0); del buf222  # reuse
    buf226 = buf197; del buf197  # reuse
    buf227 = reinterpret_tensor(buf189, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf189  # reuse
    buf228 = reinterpret_tensor(buf198, (8, 8, 50, 64), (25600, 3200, 64, 1), 0); del buf198  # reuse
    cpp_fused__softmax_clone_55(c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del buf224
    del buf225
    buf229 = buf202; del buf202  # reuse
    # Source Nodes: [factor_att_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (64, 64, 50), (3200, 1, 64), 0), reinterpret_tensor(buf228, (64, 50, 64), (3200, 64, 1), 0), out=buf229)
    buf230 = buf228; del buf228  # reuse
    cpp_fused_clone_56(c_void_p(buf223.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = reinterpret_tensor(buf227, (64, 50, 64), (3200, 64, 1), 0); del buf227  # reuse
    # Source Nodes: [factor_att_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf230, (64, 50, 64), (3200, 64, 1), 0), reinterpret_tensor(buf229, (64, 64, 64), (4096, 64, 1), 0), out=buf231)
    del buf229
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_3], Original ATen: [aten.convolution]
    buf232 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 128, 7, 7), (76800, 1, 10752, 1536), 2560), arg130_1, arg131_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128)
    assert_size_stride(buf232, (8, 128, 7, 7), (6272, 1, 896, 128))
    del arg130_1
    del arg131_1
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_4], Original ATen: [aten.convolution]
    buf233 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2688), arg132_1, arg133_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf233, (8, 192, 7, 7), (9408, 1, 1344, 192))
    del arg132_1
    del arg133_1
    # Source Nodes: [l__mod___serial_blocks4_0_factoratt_crpe_crpe_conv_list_5], Original ATen: [aten.convolution]
    buf234 = extern_kernels.convolution(reinterpret_tensor(buf223, (8, 192, 7, 7), (76800, 1, 10752, 1536), 2880), arg134_1, arg135_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=192)
    assert_size_stride(buf234, (8, 192, 7, 7), (9408, 1, 1344, 192))
    del arg134_1
    del arg135_1
    buf235 = reinterpret_tensor(buf230, (8, 50, 8, 64), (25600, 512, 64, 1), 0); del buf230  # reuse
    cpp_fused_clone_57(c_void_p(buf231.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del buf223
    del buf232
    del buf233
    del buf234
    buf236 = reinterpret_tensor(buf231, (400, 512), (512, 1), 0); del buf231  # reuse
    # Source Nodes: [x_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf235, (400, 512), (512, 1), 0), reinterpret_tensor(arg144_1, (512, 512), (1, 512), 0), alpha=1, beta=1, out=buf236)
    del arg144_1
    del arg145_1
    buf237 = buf220; del buf220  # reuse
    buf238 = buf219; del buf219  # reuse
    buf240 = reinterpret_tensor(buf235, (8, 50, 512), (25600, 512, 1), 0); del buf235  # reuse
    cpp_fused_add_cat_native_layer_norm_58(c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg34_1
    del arg35_1
    buf241 = reinterpret_tensor(buf215, (400, 2048), (2048, 1), 0); del buf215  # reuse
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg147_1, reinterpret_tensor(buf240, (400, 512), (512, 1), 0), reinterpret_tensor(arg146_1, (512, 2048), (1, 512), 0), alpha=1, beta=1, out=buf241)
    del arg146_1
    del arg147_1
    buf242 = reinterpret_tensor(buf241, (8, 50, 2048), (102400, 2048, 1), 0); del buf241  # reuse
    cpp_fused_gelu_59(c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf240, (400, 512), (512, 1), 0); del buf240  # reuse
    # Source Nodes: [x_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg149_1, reinterpret_tensor(buf242, (400, 2048), (2048, 1), 0), reinterpret_tensor(arg148_1, (2048, 512), (1, 2048), 0), alpha=1, beta=1, out=buf243)
    del arg148_1
    del arg149_1
    del buf242
    buf244 = reinterpret_tensor(buf243, (8, 50, 512), (25600, 512, 1), 0); del buf243  # reuse
    buf245 = buf238; del buf238  # reuse
    buf246 = buf237; del buf237  # reuse
    buf248 = reinterpret_tensor(buf226, (8, 512), (512, 1), 0); del buf226  # reuse
    cpp_fused_add_cat_clone_native_layer_norm_60(c_void_p(buf244.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()))
    del arg36_1
    del arg37_1
    del buf217
    del buf218
    del buf236
    del buf244
    del buf245
    del buf246
    buf249 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_162, x_163], Original ATen: [aten.addmm, aten.clone]
    extern_kernels.addmm(arg151_1, buf248, reinterpret_tensor(arg150_1, (512, 1000), (1, 512), 0), alpha=1, beta=1, out=buf249)
    del arg150_1
    del arg151_1
    return (buf249, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1, 64), (64, 64, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((1, 1, 128), (128, 128, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((1, 1, 320), (320, 320, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((1, 1, 512), (512, 512, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((64, 3, 4, 4), (48, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((192, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((24, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((192, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((64, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((512, 64), (64, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((64, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((128, 64, 2, 2), (256, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((48, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((48, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((128, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((320, 128, 2, 2), (512, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((960, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((960, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((320, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1280, 320), (320, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((320, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((512, 320, 2, 2), (1280, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((192, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((192, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1536, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((512, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((2048, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((512, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1000, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('coat_lite_mini', benchmark_compiled_module)
