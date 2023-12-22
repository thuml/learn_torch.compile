
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                }
            }
        }
    }
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
                        auto tmp0 = in_ptr1[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (8L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (8L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (16L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (8L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (8L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (8L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (8L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr4[static_cast<long>(x1 + (16L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(8);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (8L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(16);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>((-8L) + x1 + (8L*x0))];
                        auto tmp13 = out_ptr0[static_cast<long>((-8L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = out_ptr1[static_cast<long>((-8L) + x1)];
                        auto tmp16 = static_cast<float>(100352.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr2[static_cast<long>((-8L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr3[static_cast<long>((-8L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                    out_ptr3[static_cast<long>(x1 + (16L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (24L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (48L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (24L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (48L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (48L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (12L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (12L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (12L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (12L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (12L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (12L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (12L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (12L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr0[static_cast<long>(x1 + (24L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = out_ptr1[static_cast<long>(x1)];
                    auto tmp38 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(12);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (12L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(24);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr2[static_cast<long>((-12L) + x1 + (12L*x0))];
                        auto tmp13 = in_ptr3[static_cast<long>((-12L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr4[static_cast<long>((-12L) + x1)];
                        auto tmp16 = static_cast<float>(25088.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr5[static_cast<long>((-12L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr6[static_cast<long>((-12L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp32 = static_cast<float>(25088.0);
                    auto tmp33 = tmp31 / tmp32;
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = 1 / std::sqrt(tmp35);
                    auto tmp37 = decltype(tmp30)(tmp30 * tmp36);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                    auto tmp42 = decltype(tmp27)(tmp27 + tmp41);
                    out_ptr3[static_cast<long>(x1 + (24L*x0))] = tmp42;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (36L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (72L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr3[static_cast<long>(x1 + (36L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (72L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (72L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (36L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (12L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (12L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (12L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (12L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (12L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (12L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (12L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (12L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr4[static_cast<long>(x1 + (24L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(12);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (12L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(24);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>((-12L) + x1 + (12L*x0))];
                        auto tmp13 = out_ptr0[static_cast<long>((-12L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = out_ptr1[static_cast<long>((-12L) + x1)];
                        auto tmp16 = static_cast<float>(25088.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr2[static_cast<long>((-12L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr3[static_cast<long>((-12L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                    out_ptr3[static_cast<long>(x1 + (24L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (36L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (36L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (72L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(32L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr3[static_cast<long>(x1 + (36L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (72L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (36L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (36L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(36L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (36L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (72L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (36L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (72L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(72L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (20L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (20L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (20L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (20L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (20L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (20L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (20L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (20L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr0[static_cast<long>(x1 + (40L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = out_ptr1[static_cast<long>(x1)];
                    auto tmp38 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(20);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (20L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr2[static_cast<long>((-20L) + x1 + (20L*x0))];
                        auto tmp13 = in_ptr3[static_cast<long>((-20L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr4[static_cast<long>((-20L) + x1)];
                        auto tmp16 = static_cast<float>(6272.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr5[static_cast<long>((-20L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr6[static_cast<long>((-20L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp32 = static_cast<float>(6272.0);
                    auto tmp33 = tmp31 / tmp32;
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = 1 / std::sqrt(tmp35);
                    auto tmp37 = decltype(tmp30)(tmp30 * tmp36);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                    auto tmp42 = decltype(tmp27)(tmp27 + tmp41);
                    out_ptr3[static_cast<long>(x1 + (40L*x0))] = tmp42;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (60L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (60L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (60L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (60L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (120L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (60L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr3[static_cast<long>(x1 + (60L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (120L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (60L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (60L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (60L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (120L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (60L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (20L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (20L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (20L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (20L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(16L); x1<static_cast<long>(20L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (20L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    out_ptr3[static_cast<long>(x1 + (20L*x0))] = tmp13;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (20L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (20L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr4[static_cast<long>(x1 + (40L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(20);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (20L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>((-20L) + x1 + (20L*x0))];
                        auto tmp13 = out_ptr0[static_cast<long>((-20L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = out_ptr1[static_cast<long>((-20L) + x1)];
                        auto tmp16 = static_cast<float>(6272.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr2[static_cast<long>((-20L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr3[static_cast<long>((-20L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                    out_ptr3[static_cast<long>(x1 + (40L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (120L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (120L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (240L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (120L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = out_ptr1[static_cast<long>(x1)];
                    auto tmp38 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr2[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr4[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1568.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr5[static_cast<long>((-40L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr6[static_cast<long>((-40L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp32 = static_cast<float>(1568.0);
                    auto tmp33 = tmp31 / tmp32;
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = 1 / std::sqrt(tmp35);
                    auto tmp37 = decltype(tmp30)(tmp30 * tmp36);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                    auto tmp42 = decltype(tmp27)(tmp27 + tmp41);
                    out_ptr3[static_cast<long>(x1 + (80L*x0))] = tmp42;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (100L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (100L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (100L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (100L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (200L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(96L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (100L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr3[static_cast<long>(x1 + (100L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (200L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (100L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (100L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(100L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (100L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (200L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (100L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_42 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr4[static_cast<long>(x1 + (80L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = out_ptr0[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = out_ptr1[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1568.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr2[static_cast<long>((-40L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                    out_ptr3[static_cast<long>(x1 + (80L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (92L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (184L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr3[static_cast<long>(x1 + (92L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (184L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (184L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (92L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_46 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr4[static_cast<long>(x1 + (80L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = out_ptr0[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = out_ptr1[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1568.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr2[static_cast<long>((-40L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                    out_ptr3[static_cast<long>(x1 + (80L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(88L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (92L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (92L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (184L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(88L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    out_ptr3[static_cast<long>(x1 + (92L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (184L*x0))] = tmp14;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (92L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
            #pragma omp for simd simdlen(4) 
            for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0 + (92L*x1))];
                        tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                    }
                    out_ptr0[static_cast<long>(x0)] = tmp_acc0.mean;
                    out_ptr1[static_cast<long>(x0)] = tmp_acc0.m2;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1e-05);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp5 = 1 / std::sqrt(tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(92L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (92L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (184L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (92L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (40L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_50 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr4[static_cast<long>(x1 + (80L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(40);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (40L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(80);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>((-40L) + x1 + (40L*x0))];
                        auto tmp13 = out_ptr0[static_cast<long>((-40L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = out_ptr1[static_cast<long>((-40L) + x1)];
                        auto tmp16 = static_cast<float>(1568.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr2[static_cast<long>((-40L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr3[static_cast<long>((-40L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                    out_ptr3[static_cast<long>(x1 + (80L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (240L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (240L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (480L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (240L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (56L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (56L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr0[static_cast<long>(x1 + (112L*x0))];
                    auto tmp29 = out_ptr0[static_cast<long>(x1)];
                    auto tmp31 = out_ptr1[static_cast<long>(x1)];
                    auto tmp38 = in_ptr7[static_cast<long>(x1)];
                    auto tmp40 = in_ptr8[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(56);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (56L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(112);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr2[static_cast<long>((-56L) + x1 + (56L*x0))];
                        auto tmp13 = in_ptr3[static_cast<long>((-56L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = in_ptr4[static_cast<long>((-56L) + x1)];
                        auto tmp16 = static_cast<float>(1568.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr5[static_cast<long>((-56L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr6[static_cast<long>((-56L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp32 = static_cast<float>(1568.0);
                    auto tmp33 = tmp31 / tmp32;
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = 1 / std::sqrt(tmp35);
                    auto tmp37 = decltype(tmp30)(tmp30 * tmp36);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                    auto tmp42 = decltype(tmp27)(tmp27 + tmp41);
                    out_ptr3[static_cast<long>(x1 + (112L*x0))] = tmp42;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (336L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (336L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (672L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (336L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (56L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(1568.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (56L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_64 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (56L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(1568.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(1L))
                {
                    auto tmp28 = in_ptr4[static_cast<long>(x1 + (112L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(56);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr1[static_cast<long>(x1 + (56L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(112);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr0[static_cast<long>((-56L) + x1 + (56L*x0))];
                        auto tmp13 = out_ptr0[static_cast<long>((-56L) + x1)];
                        auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                        auto tmp15 = out_ptr1[static_cast<long>((-56L) + x1)];
                        auto tmp16 = static_cast<float>(1568.0);
                        auto tmp17 = tmp15 / tmp16;
                        auto tmp18 = static_cast<float>(1e-05);
                        auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                        auto tmp20 = 1 / std::sqrt(tmp19);
                        auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                        auto tmp22 = in_ptr2[static_cast<long>((-56L) + x1)];
                        auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                        auto tmp24 = in_ptr3[static_cast<long>((-56L) + x1)];
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        return tmp25;
                    }
                    ;
                    auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp27 = tmp4 ? tmp7 : tmp26;
                    auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                    out_ptr3[static_cast<long>(x1 + (112L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (336L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (336L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (336L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (336L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (672L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (336L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr4 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr4 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1344L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp28 = in_ptr0[static_cast<long>(x1 + (160L*x0))];
                auto tmp29 = out_ptr0[static_cast<long>(x1)];
                auto tmp31 = out_ptr1[static_cast<long>(x1)];
                auto tmp38 = in_ptr7[static_cast<long>(x1)];
                auto tmp40 = in_ptr8[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr1[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr2[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = in_ptr4[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(392.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                    auto tmp22 = in_ptr5[static_cast<long>((-80L) + x1)];
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = in_ptr6[static_cast<long>((-80L) + x1)];
                    auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                    return tmp25;
                }
                ;
                auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp27 = tmp4 ? tmp7 : tmp26;
                auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                auto tmp32 = static_cast<float>(392.0);
                auto tmp33 = tmp31 / tmp32;
                auto tmp34 = static_cast<float>(1e-05);
                auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                auto tmp36 = 1 / std::sqrt(tmp35);
                auto tmp37 = decltype(tmp30)(tmp30 * tmp36);
                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                auto tmp42 = decltype(tmp27)(tmp27 + tmp41);
                out_ptr3[static_cast<long>(x1 + (160L*x0))] = tmp42;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (960L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (480L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_77 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp28 = in_ptr4[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr1[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr0[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = out_ptr0[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = out_ptr1[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(392.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                    auto tmp22 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                    return tmp25;
                }
                ;
                auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp27 = tmp4 ? tmp7 : tmp26;
                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                out_ptr3[static_cast<long>(x1 + (160L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (960L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (480L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_mul_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp9 / tmp8;
            tmp10.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_83 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp28 = in_ptr4[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr1[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr0[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = out_ptr0[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = out_ptr1[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(392.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                    auto tmp22 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                    return tmp25;
                }
                ;
                auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp27 = tmp4 ? tmp7 : tmp26;
                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                out_ptr3[static_cast<long>(x1 + (160L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (960L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (480L*x0))] = tmp16;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_87 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp28 = in_ptr4[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr1[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr0[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = out_ptr0[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = out_ptr1[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(392.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                    auto tmp22 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                    return tmp25;
                }
                ;
                auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp27 = tmp4 ? tmp7 : tmp26;
                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                out_ptr3[static_cast<long>(x1 + (160L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_cat_relu_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (480L*x0)));
                    tmp16.store(out_ptr4 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       bool* out_ptr4)
{
    auto out_ptr5 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (480L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x1)];
                    auto tmp3 = out_ptr1[static_cast<long>(x1)];
                    auto tmp10 = in_ptr1[static_cast<long>(x1)];
                    auto tmp12 = in_ptr2[static_cast<long>(x1)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = tmp13 * (tmp13>0);
                    auto tmp15 = static_cast<float>(0.0);
                    auto tmp16 = tmp14 <= tmp15;
                    out_ptr3[static_cast<long>(x1 + (960L*x0))] = tmp14;
                    out_ptr4[static_cast<long>(x1 + (480L*x0))] = tmp16;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardsigmoid_hardsigmoid_backward_mul_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       bool* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
            auto tmp3 = static_cast<float>(0.0);
            auto tmp4 = max_propagate_nan(tmp2, tmp3);
            auto tmp5 = static_cast<float>(6.0);
            auto tmp6 = min_propagate_nan(tmp4, tmp5);
            auto tmp7 = tmp6 / tmp5;
            auto tmp8 = static_cast<float>(-3.0);
            auto tmp9 = tmp0 > tmp8;
            auto tmp10 = tmp0 < tmp1;
            auto tmp11 = decltype(tmp9)(tmp9 & tmp10);
            out_ptr0[static_cast<long>(x0)] = tmp7;
            out_ptr1[static_cast<long>(x0)] = tmp11;
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp2 = tmp0 * tmp1;
                        tmp2.store(out_ptr2 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_cat_93 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
            {
                auto tmp28 = in_ptr4[static_cast<long>(x1 + (160L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(80);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr1[static_cast<long>(x1 + (80L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(160);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr0[static_cast<long>((-80L) + x1 + (80L*x0))];
                    auto tmp13 = out_ptr0[static_cast<long>((-80L) + x1)];
                    auto tmp14 = decltype(tmp12)(tmp12 - tmp13);
                    auto tmp15 = out_ptr1[static_cast<long>((-80L) + x1)];
                    auto tmp16 = static_cast<float>(392.0);
                    auto tmp17 = tmp15 / tmp16;
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = 1 / std::sqrt(tmp19);
                    auto tmp21 = decltype(tmp14)(tmp14 * tmp20);
                    auto tmp22 = in_ptr2[static_cast<long>((-80L) + x1)];
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = in_ptr3[static_cast<long>((-80L) + x1)];
                    auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                    return tmp25;
                }
                ;
                auto tmp26 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp27 = tmp4 ? tmp7 : tmp26;
                auto tmp29 = decltype(tmp27)(tmp27 + tmp28);
                out_ptr3[static_cast<long>(x1 + (160L*x0))] = tmp29;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_mean_relu_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr6 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = static_cast<float>(1.0025575447570332);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp3 * tmp9;
                    auto tmp11 = static_cast<float>(0.1);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = static_cast<float>(0.9);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp13 + tmp17;
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                    tmp18.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr5 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        }
                        tmp_acc0_vec.store(out_ptr6 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_view_95 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       bool* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(10240L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = in_ptr0[static_cast<long>(x0)];
            auto tmp1 = tmp0 * (tmp0>0);
            auto tmp2 = static_cast<float>(0.0);
            auto tmp3 = tmp1 <= tmp2;
            out_ptr0[static_cast<long>(x0)] = tmp1;
            out_ptr1[static_cast<long>(x0)] = tmp3;
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_hardsigmoid_backward_threshold_backward_96 = async_compile.cpp('''
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
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const long* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const long* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const long* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const long* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const long* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const long* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const long* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const long* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const long* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const long* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const long* in_ptr49,
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const long* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const long* in_ptr57,
                       const float* in_ptr58,
                       const float* in_ptr59,
                       const float* in_ptr60,
                       const long* in_ptr61,
                       const float* in_ptr62,
                       const float* in_ptr63,
                       const float* in_ptr64,
                       const long* in_ptr65,
                       const float* in_ptr66,
                       const float* in_ptr67,
                       const float* in_ptr68,
                       const long* in_ptr69,
                       const float* in_ptr70,
                       const float* in_ptr71,
                       const float* in_ptr72,
                       const long* in_ptr73,
                       const float* in_ptr74,
                       const float* in_ptr75,
                       const float* in_ptr76,
                       const long* in_ptr77,
                       const float* in_ptr78,
                       const float* in_ptr79,
                       const float* in_ptr80,
                       const long* in_ptr81,
                       const float* in_ptr82,
                       const float* in_ptr83,
                       const float* in_ptr84,
                       const long* in_ptr85,
                       const float* in_ptr86,
                       const float* in_ptr87,
                       const float* in_ptr88,
                       const long* in_ptr89,
                       const float* in_ptr90,
                       const float* in_ptr91,
                       const float* in_ptr92,
                       const long* in_ptr93,
                       const float* in_ptr94,
                       const float* in_ptr95,
                       const float* in_ptr96,
                       const long* in_ptr97,
                       const float* in_ptr98,
                       const float* in_ptr99,
                       const float* in_ptr100,
                       const long* in_ptr101,
                       const float* in_ptr102,
                       const float* in_ptr103,
                       const float* in_ptr104,
                       const long* in_ptr105,
                       const float* in_ptr106,
                       const float* in_ptr107,
                       const float* in_ptr108,
                       const long* in_ptr109,
                       const float* in_ptr110,
                       const float* in_ptr111,
                       const float* in_ptr112,
                       const long* in_ptr113,
                       const float* in_ptr114,
                       const float* in_ptr115,
                       const float* in_ptr116,
                       const long* in_ptr117,
                       const float* in_ptr118,
                       const float* in_ptr119,
                       const float* in_ptr120,
                       const long* in_ptr121,
                       const float* in_ptr122,
                       const float* in_ptr123,
                       const float* in_ptr124,
                       const long* in_ptr125,
                       const float* in_ptr126,
                       const float* in_ptr127,
                       const float* in_ptr128,
                       const long* in_ptr129,
                       const float* in_ptr130,
                       const float* in_ptr131,
                       const float* in_ptr132,
                       const long* in_ptr133,
                       const float* in_ptr134,
                       const float* in_ptr135,
                       const float* in_ptr136,
                       const long* in_ptr137,
                       const float* in_ptr138,
                       const float* in_ptr139,
                       const float* in_ptr140,
                       const long* in_ptr141,
                       const float* in_ptr142,
                       const float* in_ptr143,
                       const float* in_ptr144,
                       const long* in_ptr145,
                       const float* in_ptr146,
                       const float* in_ptr147,
                       const float* in_ptr148,
                       const long* in_ptr149,
                       const float* in_ptr150,
                       const float* in_ptr151,
                       const float* in_ptr152,
                       const long* in_ptr153,
                       const float* in_ptr154,
                       const float* in_ptr155,
                       const float* in_ptr156,
                       const long* in_ptr157,
                       const float* in_ptr158,
                       const float* in_ptr159,
                       const float* in_ptr160,
                       const long* in_ptr161,
                       const float* in_ptr162,
                       const float* in_ptr163,
                       const float* in_ptr164,
                       const long* in_ptr165,
                       const float* in_ptr166,
                       const float* in_ptr167,
                       const float* in_ptr168,
                       const long* in_ptr169,
                       const float* in_ptr170,
                       const float* in_ptr171,
                       const float* in_ptr172,
                       const long* in_ptr173,
                       const float* in_ptr174,
                       const float* in_ptr175,
                       const float* in_ptr176,
                       const long* in_ptr177,
                       const float* in_ptr178,
                       const float* in_ptr179,
                       const float* in_ptr180,
                       const long* in_ptr181,
                       const float* in_ptr182,
                       const float* in_ptr183,
                       const float* in_ptr184,
                       const long* in_ptr185,
                       const float* in_ptr186,
                       const float* in_ptr187,
                       const float* in_ptr188,
                       const long* in_ptr189,
                       const float* in_ptr190,
                       const float* in_ptr191,
                       const float* in_ptr192,
                       const long* in_ptr193,
                       const float* in_ptr194,
                       const float* in_ptr195,
                       const float* in_ptr196,
                       const long* in_ptr197,
                       const float* in_ptr198,
                       const float* in_ptr199,
                       const float* in_ptr200,
                       const long* in_ptr201,
                       const float* in_ptr202,
                       const float* in_ptr203,
                       const float* in_ptr204,
                       const long* in_ptr205,
                       const float* in_ptr206,
                       const float* in_ptr207,
                       const float* in_ptr208,
                       const long* in_ptr209,
                       const float* in_ptr210,
                       const float* in_ptr211,
                       const float* in_ptr212,
                       const long* in_ptr213,
                       const float* in_ptr214,
                       const float* in_ptr215,
                       const float* in_ptr216,
                       const long* in_ptr217,
                       const float* in_ptr218,
                       const float* in_ptr219,
                       const float* in_ptr220,
                       const long* in_ptr221,
                       const float* in_ptr222,
                       const float* in_ptr223,
                       const float* in_ptr224,
                       const long* in_ptr225,
                       const float* in_ptr226,
                       const float* in_ptr227,
                       const float* in_ptr228,
                       const long* in_ptr229,
                       const float* in_ptr230,
                       const float* in_ptr231,
                       const float* in_ptr232,
                       const long* in_ptr233,
                       const float* in_ptr234,
                       const float* in_ptr235,
                       const float* in_ptr236,
                       const long* in_ptr237,
                       const float* in_ptr238,
                       const float* in_ptr239,
                       const float* in_ptr240,
                       const long* in_ptr241,
                       const float* in_ptr242,
                       const float* in_ptr243,
                       const float* in_ptr244,
                       const long* in_ptr245,
                       const float* in_ptr246,
                       const float* in_ptr247,
                       const float* in_ptr248,
                       const long* in_ptr249,
                       const float* in_ptr250,
                       const float* in_ptr251,
                       bool* out_ptr0,
                       bool* out_ptr1,
                       bool* out_ptr2,
                       bool* out_ptr3,
                       bool* out_ptr4,
                       bool* out_ptr5,
                       bool* out_ptr6,
                       long* out_ptr8,
                       float* out_ptr10,
                       float* out_ptr12,
                       float* out_ptr13,
                       long* out_ptr15,
                       float* out_ptr17,
                       float* out_ptr18,
                       long* out_ptr20,
                       float* out_ptr22,
                       float* out_ptr23,
                       long* out_ptr25,
                       float* out_ptr27,
                       float* out_ptr28,
                       long* out_ptr30,
                       float* out_ptr32,
                       float* out_ptr33,
                       long* out_ptr35,
                       float* out_ptr37,
                       float* out_ptr38,
                       long* out_ptr40,
                       float* out_ptr42,
                       float* out_ptr43,
                       long* out_ptr45,
                       float* out_ptr47,
                       float* out_ptr48,
                       long* out_ptr50,
                       float* out_ptr52,
                       float* out_ptr53,
                       long* out_ptr55,
                       float* out_ptr57,
                       float* out_ptr58,
                       long* out_ptr60,
                       float* out_ptr62,
                       float* out_ptr63,
                       long* out_ptr65,
                       float* out_ptr67,
                       float* out_ptr68,
                       long* out_ptr70,
                       float* out_ptr72,
                       float* out_ptr73,
                       long* out_ptr75,
                       float* out_ptr77,
                       float* out_ptr78,
                       long* out_ptr80,
                       float* out_ptr82,
                       float* out_ptr83,
                       long* out_ptr85,
                       float* out_ptr87,
                       float* out_ptr88,
                       long* out_ptr90,
                       float* out_ptr92,
                       float* out_ptr93,
                       long* out_ptr95,
                       float* out_ptr97,
                       float* out_ptr98,
                       long* out_ptr100,
                       float* out_ptr102,
                       float* out_ptr103,
                       long* out_ptr105,
                       float* out_ptr107,
                       float* out_ptr108,
                       long* out_ptr110,
                       float* out_ptr112,
                       float* out_ptr113,
                       long* out_ptr115,
                       float* out_ptr117,
                       float* out_ptr118,
                       long* out_ptr120,
                       float* out_ptr122,
                       float* out_ptr123,
                       long* out_ptr125,
                       float* out_ptr127,
                       float* out_ptr128,
                       long* out_ptr130,
                       float* out_ptr132,
                       float* out_ptr133,
                       long* out_ptr135,
                       float* out_ptr137,
                       float* out_ptr138,
                       long* out_ptr140,
                       float* out_ptr142,
                       float* out_ptr143,
                       long* out_ptr145,
                       float* out_ptr147,
                       float* out_ptr148,
                       long* out_ptr150,
                       float* out_ptr152,
                       float* out_ptr153,
                       long* out_ptr155,
                       float* out_ptr157,
                       float* out_ptr158,
                       long* out_ptr160,
                       float* out_ptr162,
                       float* out_ptr163,
                       long* out_ptr165,
                       float* out_ptr167,
                       float* out_ptr168,
                       long* out_ptr170,
                       float* out_ptr172,
                       float* out_ptr173,
                       long* out_ptr175,
                       float* out_ptr177,
                       float* out_ptr178,
                       long* out_ptr180,
                       float* out_ptr182,
                       float* out_ptr183,
                       long* out_ptr185,
                       float* out_ptr187,
                       float* out_ptr188,
                       long* out_ptr190,
                       float* out_ptr192,
                       float* out_ptr193,
                       long* out_ptr195,
                       float* out_ptr197,
                       float* out_ptr198,
                       long* out_ptr200,
                       float* out_ptr202,
                       float* out_ptr203,
                       long* out_ptr205,
                       float* out_ptr207,
                       float* out_ptr208,
                       long* out_ptr210,
                       float* out_ptr212,
                       float* out_ptr213,
                       long* out_ptr215,
                       float* out_ptr217,
                       float* out_ptr218,
                       long* out_ptr220,
                       float* out_ptr222,
                       float* out_ptr223,
                       long* out_ptr225,
                       float* out_ptr227,
                       float* out_ptr228,
                       long* out_ptr230,
                       float* out_ptr232,
                       float* out_ptr233,
                       long* out_ptr235,
                       float* out_ptr237,
                       float* out_ptr238,
                       long* out_ptr240,
                       float* out_ptr242,
                       float* out_ptr243,
                       long* out_ptr245,
                       float* out_ptr247,
                       float* out_ptr248,
                       long* out_ptr250,
                       float* out_ptr252,
                       float* out_ptr253,
                       long* out_ptr255,
                       float* out_ptr257,
                       float* out_ptr258,
                       long* out_ptr260,
                       float* out_ptr262,
                       float* out_ptr263,
                       long* out_ptr265,
                       float* out_ptr267,
                       float* out_ptr268,
                       long* out_ptr270,
                       float* out_ptr272,
                       float* out_ptr273,
                       long* out_ptr275,
                       float* out_ptr277,
                       float* out_ptr278,
                       long* out_ptr280,
                       float* out_ptr282,
                       float* out_ptr283,
                       long* out_ptr285,
                       float* out_ptr287,
                       float* out_ptr288,
                       long* out_ptr290,
                       float* out_ptr292,
                       float* out_ptr293,
                       long* out_ptr295,
                       float* out_ptr297,
                       float* out_ptr298,
                       long* out_ptr300,
                       float* out_ptr302,
                       float* out_ptr303,
                       long* out_ptr305,
                       float* out_ptr307,
                       float* out_ptr308,
                       long* out_ptr310,
                       float* out_ptr312)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(376320L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = tmp0 <= tmp1;
                out_ptr0[static_cast<long>(x0)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = tmp0 > tmp1;
                    auto tmp3 = static_cast<float>(3.0);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                    out_ptr1[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr2[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = tmp0 > tmp1;
                    auto tmp3 = static_cast<float>(3.0);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                    out_ptr2[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(5376L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = tmp0 > tmp1;
                    auto tmp3 = static_cast<float>(3.0);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                    out_ptr3[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = tmp0 > tmp1;
                    auto tmp3 = static_cast<float>(3.0);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                    out_ptr4[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr5[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = tmp0 > tmp1;
                    auto tmp3 = static_cast<float>(3.0);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                    out_ptr5[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(576L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr6[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = tmp0 > tmp1;
                    auto tmp3 = static_cast<float>(3.0);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                    out_ptr6[static_cast<long>(x0)] = tmp5;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr7[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr8[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr10 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr12 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr12 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr13 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr13 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr13[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr15[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr17 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr17 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr18 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr18 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr17[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr20[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr22 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr22 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr23 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr23 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr21[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr25[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr27 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr27 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr28 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr28 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr25[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr30[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr32 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr32 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr33 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr33 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr29[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr35[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr37 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr37 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr38 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr38 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr33[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr40[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr34 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr42 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr42 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr43 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr43 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr37[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr45[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr38 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr47 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr47 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr48 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr48 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr41[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr50[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr42 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr52 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr52 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr42[static_cast<long>(x0)];
                    auto tmp3 = in_ptr43[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr52[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr53 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr53 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr8[static_cast<long>(x0)];
                    auto tmp7 = in_ptr44[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr53[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr45[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr55[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr57 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr57 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr46[static_cast<long>(x0)];
                    auto tmp3 = in_ptr47[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr57[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr58 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr58 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr9[static_cast<long>(x0)];
                    auto tmp7 = in_ptr48[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr58[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr49[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr60[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr50 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr62 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr62 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr63 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr63 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr53[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr65[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr54 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr67 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr67 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr68 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr68 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr57[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr70[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr58 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr72 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr72 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr58[static_cast<long>(x0)];
                    auto tmp3 = in_ptr59[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr72[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr73 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr73 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr12[static_cast<long>(x0)];
                    auto tmp7 = in_ptr60[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr73[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr61[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr75[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr62 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr77 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr77 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr62[static_cast<long>(x0)];
                    auto tmp3 = in_ptr63[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr77[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr78 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr78 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr13[static_cast<long>(x0)];
                    auto tmp7 = in_ptr64[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr78[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr65[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr80[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr66 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr82 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr82 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr66[static_cast<long>(x0)];
                    auto tmp3 = in_ptr67[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr82[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr83 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr83 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr14[static_cast<long>(x0)];
                    auto tmp7 = in_ptr68[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr83[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr69[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr85[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr70 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr87 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr87 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr70[static_cast<long>(x0)];
                    auto tmp3 = in_ptr71[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr87[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr88 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr88 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(8L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr15[static_cast<long>(x0)];
                    auto tmp7 = in_ptr72[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr88[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr73[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr90[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr74 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr92 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr92 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr74[static_cast<long>(x0)];
                    auto tmp3 = in_ptr75[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr92[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr93 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr93 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr16[static_cast<long>(x0)];
                    auto tmp7 = in_ptr76[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr93[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr77[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr95[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr78 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr97 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr97 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr78[static_cast<long>(x0)];
                    auto tmp3 = in_ptr79[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr97[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr98 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr98 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(32L); x0<static_cast<long>(36L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr17[static_cast<long>(x0)];
                    auto tmp7 = in_ptr80[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0000398612827361);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr98[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr81[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr100[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr82 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr102 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr102 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr103 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr103 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr85[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr105[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr86 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr107 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr107 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr86[static_cast<long>(x0)];
                    auto tmp3 = in_ptr87[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr107[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr108 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr108 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr19[static_cast<long>(x0)];
                    auto tmp7 = in_ptr88[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr108[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr89[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr110[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr90 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr112 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr112 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr90[static_cast<long>(x0)];
                    auto tmp3 = in_ptr91[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr112[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr113 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr113 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr20[static_cast<long>(x0)];
                    auto tmp7 = in_ptr92[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr113[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr93[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr115[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr94 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr117 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr117 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr118 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr118 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr97[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr120[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr98 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr122 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr122 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr123 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr123 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr101[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr125[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr102 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr127 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr127 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr102[static_cast<long>(x0)];
                    auto tmp3 = in_ptr103[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr127[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr128 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr128 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr23[static_cast<long>(x0)];
                    auto tmp7 = in_ptr104[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr128[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr105[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr130[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr106 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr132 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr132 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr106[static_cast<long>(x0)];
                    auto tmp3 = in_ptr107[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr132[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr133 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr133 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(56L); x0<static_cast<long>(60L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr24[static_cast<long>(x0)];
                    auto tmp7 = in_ptr108[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr133[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr109[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr135[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr110 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr137 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr137 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr110[static_cast<long>(x0)];
                    auto tmp3 = in_ptr111[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr137[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr138 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr138 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr25[static_cast<long>(x0)];
                    auto tmp7 = in_ptr112[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr138[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr113[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr140[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr114 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr142 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr142 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr114[static_cast<long>(x0)];
                    auto tmp3 = in_ptr115[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr142[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr143 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr143 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(16L); x0<static_cast<long>(20L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr26[static_cast<long>(x0)];
                    auto tmp7 = in_ptr116[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0001594642002871);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr143[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr117[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr145[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr118 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr147 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr147 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr148 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr148 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr121[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr150[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr122 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr152 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr152 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr153 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr153 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr125[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr155[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr126 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr157 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr157 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr158 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr158 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr129[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr160[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr130 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr162 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr162 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr163 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr163 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr133[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr165[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr134 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr167 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr167 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr168 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr168 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr137[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr170[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr138 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr172 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr172 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr173 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr173 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr141[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr175[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr142 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr177 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr177 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr178 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr178 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr145[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr180[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr146 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr182 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr182 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr146[static_cast<long>(x0)];
                    auto tmp3 = in_ptr147[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr182[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr183 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr183 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr34[static_cast<long>(x0)];
                    auto tmp7 = in_ptr148[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr183[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr149[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr185[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr150 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr187 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr187 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr150[static_cast<long>(x0)];
                    auto tmp3 = in_ptr151[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr187[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr188 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr188 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(96L); x0<static_cast<long>(100L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr35[static_cast<long>(x0)];
                    auto tmp7 = in_ptr152[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr188[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr153[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr190[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr154 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr192 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr192 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr193 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr193 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr157[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr195[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr158 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr197 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr197 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr198 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr198 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr161[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr200[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr162 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr202 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr202 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr162[static_cast<long>(x0)];
                    auto tmp3 = in_ptr163[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr202[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr203 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr203 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr38[static_cast<long>(x0)];
                    auto tmp7 = in_ptr164[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr203[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr165[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr205[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr166 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr207 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr207 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr166[static_cast<long>(x0)];
                    auto tmp3 = in_ptr167[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr207[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr208 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr208 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr39[static_cast<long>(x0)];
                    auto tmp7 = in_ptr168[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr208[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr169[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr210[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr170 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr212 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr212 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr213 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr213 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr173[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr215[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr174 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr217 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr217 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr218 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr218 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr177[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr220[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr178 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr222 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr222 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr178[static_cast<long>(x0)];
                    auto tmp3 = in_ptr179[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr222[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr223 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr223 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr42[static_cast<long>(x0)];
                    auto tmp7 = in_ptr180[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr223[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr181[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr225[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr182 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr227 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr227 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr182[static_cast<long>(x0)];
                    auto tmp3 = in_ptr183[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = static_cast<float>(0.9);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp2)(tmp2 + tmp5);
                    out_ptr227[static_cast<long>(x0)] = tmp6;
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(88L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr228 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr228 + static_cast<long>(x0));
                }
                #pragma omp simd simdlen(4) 
                for(long x0=static_cast<long>(88L); x0<static_cast<long>(92L); x0+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr43[static_cast<long>(x0)];
                    auto tmp7 = in_ptr184[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = tmp0 / tmp1;
                    auto tmp3 = static_cast<float>(1.0006381620931717);
                    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                    auto tmp5 = static_cast<float>(0.1);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp8 = static_cast<float>(0.9);
                    auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    out_ptr228[static_cast<long>(x0)] = tmp10;
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr185[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr230[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr186 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr232 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr232 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr233 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr233 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr189[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr235[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr190 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr237 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr237 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr238 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr238 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr193[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr240[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr194 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr242 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr242 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr243 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr243 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr197[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr245[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr198 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr247 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr247 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr248 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr248 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr201[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr250[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr202 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr252 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr252 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr253 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr253 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr205[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr255[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr206 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr257 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr257 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr258 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr258 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr209[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr260[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr210 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr262 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr262 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr263 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr263 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr213[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr265[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr214 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr267 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr267 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr268 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr268 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr217[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr270[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr218 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr272 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr272 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr273 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr273 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr221[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr275[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr222 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr277 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr277 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr278 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr278 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr225[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr280[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr226 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr282 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr282 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr283 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr283 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr229[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr285[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr230 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr287 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr287 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr288 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr288 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr233[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr290[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr234 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr292 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr292 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr293 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr293 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr237[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr295[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr238 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr297 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr297 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(336L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr298 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr298 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr241[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr300[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr242 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr302 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr302 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr303 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr303 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr245[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr305[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr246 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr307 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr307 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr308 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr308 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr249[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr310[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr250 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr312 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr312 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_97 = async_compile.cpp('''
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
                       const float* in_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const long* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const long* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const long* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const long* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const long* in_ptr21,
                       const float* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const long* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const long* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const long* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const long* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const long* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const long* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const long* in_ptr49,
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const long* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const long* in_ptr57,
                       const float* in_ptr58,
                       const float* in_ptr59,
                       const float* in_ptr60,
                       const long* in_ptr61,
                       const float* in_ptr62,
                       const float* in_ptr63,
                       const float* in_ptr64,
                       const long* in_ptr65,
                       const float* in_ptr66,
                       const float* in_ptr67,
                       const float* in_ptr68,
                       const long* in_ptr69,
                       const float* in_ptr70,
                       const float* in_ptr71,
                       const float* in_ptr72,
                       const long* in_ptr73,
                       float* out_ptr0,
                       long* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       long* out_ptr7,
                       float* out_ptr9,
                       float* out_ptr10,
                       long* out_ptr12,
                       float* out_ptr14,
                       float* out_ptr15,
                       long* out_ptr17,
                       float* out_ptr19,
                       float* out_ptr20,
                       long* out_ptr22,
                       float* out_ptr24,
                       float* out_ptr25,
                       long* out_ptr27,
                       float* out_ptr29,
                       float* out_ptr30,
                       long* out_ptr32,
                       float* out_ptr34,
                       float* out_ptr35,
                       long* out_ptr37,
                       float* out_ptr39,
                       float* out_ptr40,
                       long* out_ptr42,
                       float* out_ptr44,
                       float* out_ptr45,
                       long* out_ptr47,
                       float* out_ptr49,
                       float* out_ptr50,
                       long* out_ptr52,
                       float* out_ptr54,
                       float* out_ptr55,
                       long* out_ptr57,
                       float* out_ptr59,
                       float* out_ptr60,
                       long* out_ptr62,
                       float* out_ptr64,
                       float* out_ptr65,
                       long* out_ptr67,
                       float* out_ptr69,
                       float* out_ptr70,
                       long* out_ptr72,
                       float* out_ptr74,
                       float* out_ptr75,
                       long* out_ptr77,
                       float* out_ptr79,
                       float* out_ptr80,
                       long* out_ptr82,
                       float* out_ptr84,
                       float* out_ptr85,
                       long* out_ptr87,
                       float* out_ptr89,
                       float* out_ptr90,
                       long* out_ptr92)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr1[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr2[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr4 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr5 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr5[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr7[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr9 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr9 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr10 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr10 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr9[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr12[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr14 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr14 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr15 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr15 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr13[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr17[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr19 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr19 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr20 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr20 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr17[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr22[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr24 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr24 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr25 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr25 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr21[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr27[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr29 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr29 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr30 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr30 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr25[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr32[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr34 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr34 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr35 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr35 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr29[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr37[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr39 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr39 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr40 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr40 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr33[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr42[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr34 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr44 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr44 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr45 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr45 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr37[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr47[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr38 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr49 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr49 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr50 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr50 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr41[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr52[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr42 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr54 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr54 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr55 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr55 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr45[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr57[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr59 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr59 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr60 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr60 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr49[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr62[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr50 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr64 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr64 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr65 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr65 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr53[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr67[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr54 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr69 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr69 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr70 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr70 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr57[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr72[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr58 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr74 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr74 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr75 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr75 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr61[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr77[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr62 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr79 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr79 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr80 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr80 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr65[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr82[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr66 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr84 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr84 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr85 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr85 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr69[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr87[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr70 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr89 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr89 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr90 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0025575447570332);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr90 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr73[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr92[static_cast<long>(0L)] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513 = args
    args.clear()
    assert_size_stride(primals_1, (960, ), (1, ))
    assert_size_stride(primals_2, (960, ), (1, ))
    assert_size_stride(primals_3, (1000, 1280), (1280, 1))
    assert_size_stride(primals_4, (1000, ), (1, ))
    assert_size_stride(primals_5, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_9, (8, ), (1, ))
    assert_size_stride(primals_10, (8, ), (1, ))
    assert_size_stride(primals_11, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_12, (8, ), (1, ))
    assert_size_stride(primals_13, (8, ), (1, ))
    assert_size_stride(primals_14, (8, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_15, (8, ), (1, ))
    assert_size_stride(primals_16, (8, ), (1, ))
    assert_size_stride(primals_17, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_18, (8, ), (1, ))
    assert_size_stride(primals_19, (8, ), (1, ))
    assert_size_stride(primals_20, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_22, (24, ), (1, ))
    assert_size_stride(primals_23, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (24, ), (1, ))
    assert_size_stride(primals_26, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (48, ), (1, ))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (12, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_30, (12, ), (1, ))
    assert_size_stride(primals_31, (12, ), (1, ))
    assert_size_stride(primals_32, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_33, (12, ), (1, ))
    assert_size_stride(primals_34, (12, ), (1, ))
    assert_size_stride(primals_35, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_36, (16, ), (1, ))
    assert_size_stride(primals_37, (16, ), (1, ))
    assert_size_stride(primals_38, (24, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_39, (24, ), (1, ))
    assert_size_stride(primals_40, (24, ), (1, ))
    assert_size_stride(primals_41, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_42, (36, ), (1, ))
    assert_size_stride(primals_43, (36, ), (1, ))
    assert_size_stride(primals_44, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_45, (36, ), (1, ))
    assert_size_stride(primals_46, (36, ), (1, ))
    assert_size_stride(primals_47, (12, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_48, (12, ), (1, ))
    assert_size_stride(primals_49, (12, ), (1, ))
    assert_size_stride(primals_50, (12, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (12, ), (1, ))
    assert_size_stride(primals_52, (12, ), (1, ))
    assert_size_stride(primals_53, (36, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_54, (36, ), (1, ))
    assert_size_stride(primals_55, (36, ), (1, ))
    assert_size_stride(primals_56, (36, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_57, (36, ), (1, ))
    assert_size_stride(primals_58, (36, ), (1, ))
    assert_size_stride(primals_59, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_60, (72, ), (1, ))
    assert_size_stride(primals_61, (72, ), (1, ))
    assert_size_stride(primals_62, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_63, (20, ), (1, ))
    assert_size_stride(primals_64, (72, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(primals_65, (72, ), (1, ))
    assert_size_stride(primals_66, (20, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_67, (20, ), (1, ))
    assert_size_stride(primals_68, (20, ), (1, ))
    assert_size_stride(primals_69, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_70, (20, ), (1, ))
    assert_size_stride(primals_71, (20, ), (1, ))
    assert_size_stride(primals_72, (24, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_73, (24, ), (1, ))
    assert_size_stride(primals_74, (24, ), (1, ))
    assert_size_stride(primals_75, (40, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_76, (40, ), (1, ))
    assert_size_stride(primals_77, (40, ), (1, ))
    assert_size_stride(primals_78, (60, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_79, (60, ), (1, ))
    assert_size_stride(primals_80, (60, ), (1, ))
    assert_size_stride(primals_81, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_82, (60, ), (1, ))
    assert_size_stride(primals_83, (60, ), (1, ))
    assert_size_stride(primals_84, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_86, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_87, (120, ), (1, ))
    assert_size_stride(primals_88, (20, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_89, (20, ), (1, ))
    assert_size_stride(primals_90, (20, ), (1, ))
    assert_size_stride(primals_91, (20, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_92, (20, ), (1, ))
    assert_size_stride(primals_93, (20, ), (1, ))
    assert_size_stride(primals_94, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_95, (120, ), (1, ))
    assert_size_stride(primals_96, (120, ), (1, ))
    assert_size_stride(primals_97, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (120, ), (1, ))
    assert_size_stride(primals_99, (120, ), (1, ))
    assert_size_stride(primals_100, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_101, (240, ), (1, ))
    assert_size_stride(primals_102, (240, ), (1, ))
    assert_size_stride(primals_103, (40, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_104, (40, ), (1, ))
    assert_size_stride(primals_105, (40, ), (1, ))
    assert_size_stride(primals_106, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (40, ), (1, ))
    assert_size_stride(primals_108, (40, ), (1, ))
    assert_size_stride(primals_109, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_110, (40, ), (1, ))
    assert_size_stride(primals_111, (40, ), (1, ))
    assert_size_stride(primals_112, (80, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_113, (80, ), (1, ))
    assert_size_stride(primals_114, (80, ), (1, ))
    assert_size_stride(primals_115, (100, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_116, (100, ), (1, ))
    assert_size_stride(primals_117, (100, ), (1, ))
    assert_size_stride(primals_118, (100, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_119, (100, ), (1, ))
    assert_size_stride(primals_120, (100, ), (1, ))
    assert_size_stride(primals_121, (40, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_122, (40, ), (1, ))
    assert_size_stride(primals_123, (40, ), (1, ))
    assert_size_stride(primals_124, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_125, (40, ), (1, ))
    assert_size_stride(primals_126, (40, ), (1, ))
    assert_size_stride(primals_127, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_128, (92, ), (1, ))
    assert_size_stride(primals_129, (92, ), (1, ))
    assert_size_stride(primals_130, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_131, (92, ), (1, ))
    assert_size_stride(primals_132, (92, ), (1, ))
    assert_size_stride(primals_133, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_134, (40, ), (1, ))
    assert_size_stride(primals_135, (40, ), (1, ))
    assert_size_stride(primals_136, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_137, (40, ), (1, ))
    assert_size_stride(primals_138, (40, ), (1, ))
    assert_size_stride(primals_139, (92, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_140, (92, ), (1, ))
    assert_size_stride(primals_141, (92, ), (1, ))
    assert_size_stride(primals_142, (92, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (92, ), (1, ))
    assert_size_stride(primals_144, (92, ), (1, ))
    assert_size_stride(primals_145, (40, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_146, (40, ), (1, ))
    assert_size_stride(primals_147, (40, ), (1, ))
    assert_size_stride(primals_148, (40, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (40, ), (1, ))
    assert_size_stride(primals_150, (40, ), (1, ))
    assert_size_stride(primals_151, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_152, (240, ), (1, ))
    assert_size_stride(primals_153, (240, ), (1, ))
    assert_size_stride(primals_154, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (240, ), (1, ))
    assert_size_stride(primals_156, (240, ), (1, ))
    assert_size_stride(primals_157, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_158, (120, ), (1, ))
    assert_size_stride(primals_159, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_160, (480, ), (1, ))
    assert_size_stride(primals_161, (56, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_162, (56, ), (1, ))
    assert_size_stride(primals_163, (56, ), (1, ))
    assert_size_stride(primals_164, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_165, (56, ), (1, ))
    assert_size_stride(primals_166, (56, ), (1, ))
    assert_size_stride(primals_167, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (80, ), (1, ))
    assert_size_stride(primals_169, (80, ), (1, ))
    assert_size_stride(primals_170, (112, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_171, (112, ), (1, ))
    assert_size_stride(primals_172, (112, ), (1, ))
    assert_size_stride(primals_173, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_174, (336, ), (1, ))
    assert_size_stride(primals_175, (336, ), (1, ))
    assert_size_stride(primals_176, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_177, (336, ), (1, ))
    assert_size_stride(primals_178, (336, ), (1, ))
    assert_size_stride(primals_179, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_180, (168, ), (1, ))
    assert_size_stride(primals_181, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_182, (672, ), (1, ))
    assert_size_stride(primals_183, (56, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_184, (56, ), (1, ))
    assert_size_stride(primals_185, (56, ), (1, ))
    assert_size_stride(primals_186, (56, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_187, (56, ), (1, ))
    assert_size_stride(primals_188, (56, ), (1, ))
    assert_size_stride(primals_189, (336, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_190, (336, ), (1, ))
    assert_size_stride(primals_191, (336, ), (1, ))
    assert_size_stride(primals_192, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (336, ), (1, ))
    assert_size_stride(primals_194, (336, ), (1, ))
    assert_size_stride(primals_195, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_196, (672, ), (1, ))
    assert_size_stride(primals_197, (672, ), (1, ))
    assert_size_stride(primals_198, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_199, (168, ), (1, ))
    assert_size_stride(primals_200, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_201, (672, ), (1, ))
    assert_size_stride(primals_202, (80, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_203, (80, ), (1, ))
    assert_size_stride(primals_204, (80, ), (1, ))
    assert_size_stride(primals_205, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_206, (80, ), (1, ))
    assert_size_stride(primals_207, (80, ), (1, ))
    assert_size_stride(primals_208, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_209, (112, ), (1, ))
    assert_size_stride(primals_210, (112, ), (1, ))
    assert_size_stride(primals_211, (160, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_212, (160, ), (1, ))
    assert_size_stride(primals_213, (160, ), (1, ))
    assert_size_stride(primals_214, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_215, (480, ), (1, ))
    assert_size_stride(primals_216, (480, ), (1, ))
    assert_size_stride(primals_217, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (480, ), (1, ))
    assert_size_stride(primals_219, (480, ), (1, ))
    assert_size_stride(primals_220, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_221, (80, ), (1, ))
    assert_size_stride(primals_222, (80, ), (1, ))
    assert_size_stride(primals_223, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_224, (80, ), (1, ))
    assert_size_stride(primals_225, (80, ), (1, ))
    assert_size_stride(primals_226, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_227, (480, ), (1, ))
    assert_size_stride(primals_228, (480, ), (1, ))
    assert_size_stride(primals_229, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_230, (480, ), (1, ))
    assert_size_stride(primals_231, (480, ), (1, ))
    assert_size_stride(primals_232, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_233, (240, ), (1, ))
    assert_size_stride(primals_234, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_235, (960, ), (1, ))
    assert_size_stride(primals_236, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_237, (80, ), (1, ))
    assert_size_stride(primals_238, (80, ), (1, ))
    assert_size_stride(primals_239, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_240, (80, ), (1, ))
    assert_size_stride(primals_241, (80, ), (1, ))
    assert_size_stride(primals_242, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_243, (480, ), (1, ))
    assert_size_stride(primals_244, (480, ), (1, ))
    assert_size_stride(primals_245, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (480, ), (1, ))
    assert_size_stride(primals_247, (480, ), (1, ))
    assert_size_stride(primals_248, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_249, (80, ), (1, ))
    assert_size_stride(primals_250, (80, ), (1, ))
    assert_size_stride(primals_251, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_252, (80, ), (1, ))
    assert_size_stride(primals_253, (80, ), (1, ))
    assert_size_stride(primals_254, (480, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_255, (480, ), (1, ))
    assert_size_stride(primals_256, (480, ), (1, ))
    assert_size_stride(primals_257, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_258, (480, ), (1, ))
    assert_size_stride(primals_259, (480, ), (1, ))
    assert_size_stride(primals_260, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_261, (240, ), (1, ))
    assert_size_stride(primals_262, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_263, (960, ), (1, ))
    assert_size_stride(primals_264, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_265, (80, ), (1, ))
    assert_size_stride(primals_266, (80, ), (1, ))
    assert_size_stride(primals_267, (80, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_268, (80, ), (1, ))
    assert_size_stride(primals_269, (80, ), (1, ))
    assert_size_stride(primals_270, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_271, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_272, (1280, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (960, ), (1, ))
    assert_size_stride(primals_275, (960, ), (1, ))
    assert_size_stride(primals_276, (16, ), (1, ))
    assert_size_stride(primals_277, (16, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (8, ), (1, ))
    assert_size_stride(primals_280, (8, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (8, ), (1, ))
    assert_size_stride(primals_283, (8, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (8, ), (1, ))
    assert_size_stride(primals_286, (8, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (8, ), (1, ))
    assert_size_stride(primals_289, (8, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (24, ), (1, ))
    assert_size_stride(primals_292, (24, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (24, ), (1, ))
    assert_size_stride(primals_295, (24, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (48, ), (1, ))
    assert_size_stride(primals_298, (48, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (12, ), (1, ))
    assert_size_stride(primals_301, (12, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (12, ), (1, ))
    assert_size_stride(primals_304, (12, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (16, ), (1, ))
    assert_size_stride(primals_307, (16, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (24, ), (1, ))
    assert_size_stride(primals_310, (24, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (36, ), (1, ))
    assert_size_stride(primals_313, (36, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (36, ), (1, ))
    assert_size_stride(primals_316, (36, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (12, ), (1, ))
    assert_size_stride(primals_319, (12, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (12, ), (1, ))
    assert_size_stride(primals_322, (12, ), (1, ))
    assert_size_stride(primals_323, (), ())
    assert_size_stride(primals_324, (36, ), (1, ))
    assert_size_stride(primals_325, (36, ), (1, ))
    assert_size_stride(primals_326, (), ())
    assert_size_stride(primals_327, (36, ), (1, ))
    assert_size_stride(primals_328, (36, ), (1, ))
    assert_size_stride(primals_329, (), ())
    assert_size_stride(primals_330, (72, ), (1, ))
    assert_size_stride(primals_331, (72, ), (1, ))
    assert_size_stride(primals_332, (), ())
    assert_size_stride(primals_333, (20, ), (1, ))
    assert_size_stride(primals_334, (20, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (20, ), (1, ))
    assert_size_stride(primals_337, (20, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (24, ), (1, ))
    assert_size_stride(primals_340, (24, ), (1, ))
    assert_size_stride(primals_341, (), ())
    assert_size_stride(primals_342, (40, ), (1, ))
    assert_size_stride(primals_343, (40, ), (1, ))
    assert_size_stride(primals_344, (), ())
    assert_size_stride(primals_345, (60, ), (1, ))
    assert_size_stride(primals_346, (60, ), (1, ))
    assert_size_stride(primals_347, (), ())
    assert_size_stride(primals_348, (60, ), (1, ))
    assert_size_stride(primals_349, (60, ), (1, ))
    assert_size_stride(primals_350, (), ())
    assert_size_stride(primals_351, (20, ), (1, ))
    assert_size_stride(primals_352, (20, ), (1, ))
    assert_size_stride(primals_353, (), ())
    assert_size_stride(primals_354, (20, ), (1, ))
    assert_size_stride(primals_355, (20, ), (1, ))
    assert_size_stride(primals_356, (), ())
    assert_size_stride(primals_357, (120, ), (1, ))
    assert_size_stride(primals_358, (120, ), (1, ))
    assert_size_stride(primals_359, (), ())
    assert_size_stride(primals_360, (120, ), (1, ))
    assert_size_stride(primals_361, (120, ), (1, ))
    assert_size_stride(primals_362, (), ())
    assert_size_stride(primals_363, (240, ), (1, ))
    assert_size_stride(primals_364, (240, ), (1, ))
    assert_size_stride(primals_365, (), ())
    assert_size_stride(primals_366, (40, ), (1, ))
    assert_size_stride(primals_367, (40, ), (1, ))
    assert_size_stride(primals_368, (), ())
    assert_size_stride(primals_369, (40, ), (1, ))
    assert_size_stride(primals_370, (40, ), (1, ))
    assert_size_stride(primals_371, (), ())
    assert_size_stride(primals_372, (40, ), (1, ))
    assert_size_stride(primals_373, (40, ), (1, ))
    assert_size_stride(primals_374, (), ())
    assert_size_stride(primals_375, (80, ), (1, ))
    assert_size_stride(primals_376, (80, ), (1, ))
    assert_size_stride(primals_377, (), ())
    assert_size_stride(primals_378, (100, ), (1, ))
    assert_size_stride(primals_379, (100, ), (1, ))
    assert_size_stride(primals_380, (), ())
    assert_size_stride(primals_381, (100, ), (1, ))
    assert_size_stride(primals_382, (100, ), (1, ))
    assert_size_stride(primals_383, (), ())
    assert_size_stride(primals_384, (40, ), (1, ))
    assert_size_stride(primals_385, (40, ), (1, ))
    assert_size_stride(primals_386, (), ())
    assert_size_stride(primals_387, (40, ), (1, ))
    assert_size_stride(primals_388, (40, ), (1, ))
    assert_size_stride(primals_389, (), ())
    assert_size_stride(primals_390, (92, ), (1, ))
    assert_size_stride(primals_391, (92, ), (1, ))
    assert_size_stride(primals_392, (), ())
    assert_size_stride(primals_393, (92, ), (1, ))
    assert_size_stride(primals_394, (92, ), (1, ))
    assert_size_stride(primals_395, (), ())
    assert_size_stride(primals_396, (40, ), (1, ))
    assert_size_stride(primals_397, (40, ), (1, ))
    assert_size_stride(primals_398, (), ())
    assert_size_stride(primals_399, (40, ), (1, ))
    assert_size_stride(primals_400, (40, ), (1, ))
    assert_size_stride(primals_401, (), ())
    assert_size_stride(primals_402, (92, ), (1, ))
    assert_size_stride(primals_403, (92, ), (1, ))
    assert_size_stride(primals_404, (), ())
    assert_size_stride(primals_405, (92, ), (1, ))
    assert_size_stride(primals_406, (92, ), (1, ))
    assert_size_stride(primals_407, (), ())
    assert_size_stride(primals_408, (40, ), (1, ))
    assert_size_stride(primals_409, (40, ), (1, ))
    assert_size_stride(primals_410, (), ())
    assert_size_stride(primals_411, (40, ), (1, ))
    assert_size_stride(primals_412, (40, ), (1, ))
    assert_size_stride(primals_413, (), ())
    assert_size_stride(primals_414, (240, ), (1, ))
    assert_size_stride(primals_415, (240, ), (1, ))
    assert_size_stride(primals_416, (), ())
    assert_size_stride(primals_417, (240, ), (1, ))
    assert_size_stride(primals_418, (240, ), (1, ))
    assert_size_stride(primals_419, (), ())
    assert_size_stride(primals_420, (56, ), (1, ))
    assert_size_stride(primals_421, (56, ), (1, ))
    assert_size_stride(primals_422, (), ())
    assert_size_stride(primals_423, (56, ), (1, ))
    assert_size_stride(primals_424, (56, ), (1, ))
    assert_size_stride(primals_425, (), ())
    assert_size_stride(primals_426, (80, ), (1, ))
    assert_size_stride(primals_427, (80, ), (1, ))
    assert_size_stride(primals_428, (), ())
    assert_size_stride(primals_429, (112, ), (1, ))
    assert_size_stride(primals_430, (112, ), (1, ))
    assert_size_stride(primals_431, (), ())
    assert_size_stride(primals_432, (336, ), (1, ))
    assert_size_stride(primals_433, (336, ), (1, ))
    assert_size_stride(primals_434, (), ())
    assert_size_stride(primals_435, (336, ), (1, ))
    assert_size_stride(primals_436, (336, ), (1, ))
    assert_size_stride(primals_437, (), ())
    assert_size_stride(primals_438, (56, ), (1, ))
    assert_size_stride(primals_439, (56, ), (1, ))
    assert_size_stride(primals_440, (), ())
    assert_size_stride(primals_441, (56, ), (1, ))
    assert_size_stride(primals_442, (56, ), (1, ))
    assert_size_stride(primals_443, (), ())
    assert_size_stride(primals_444, (336, ), (1, ))
    assert_size_stride(primals_445, (336, ), (1, ))
    assert_size_stride(primals_446, (), ())
    assert_size_stride(primals_447, (336, ), (1, ))
    assert_size_stride(primals_448, (336, ), (1, ))
    assert_size_stride(primals_449, (), ())
    assert_size_stride(primals_450, (672, ), (1, ))
    assert_size_stride(primals_451, (672, ), (1, ))
    assert_size_stride(primals_452, (), ())
    assert_size_stride(primals_453, (80, ), (1, ))
    assert_size_stride(primals_454, (80, ), (1, ))
    assert_size_stride(primals_455, (), ())
    assert_size_stride(primals_456, (80, ), (1, ))
    assert_size_stride(primals_457, (80, ), (1, ))
    assert_size_stride(primals_458, (), ())
    assert_size_stride(primals_459, (112, ), (1, ))
    assert_size_stride(primals_460, (112, ), (1, ))
    assert_size_stride(primals_461, (), ())
    assert_size_stride(primals_462, (160, ), (1, ))
    assert_size_stride(primals_463, (160, ), (1, ))
    assert_size_stride(primals_464, (), ())
    assert_size_stride(primals_465, (480, ), (1, ))
    assert_size_stride(primals_466, (480, ), (1, ))
    assert_size_stride(primals_467, (), ())
    assert_size_stride(primals_468, (480, ), (1, ))
    assert_size_stride(primals_469, (480, ), (1, ))
    assert_size_stride(primals_470, (), ())
    assert_size_stride(primals_471, (80, ), (1, ))
    assert_size_stride(primals_472, (80, ), (1, ))
    assert_size_stride(primals_473, (), ())
    assert_size_stride(primals_474, (80, ), (1, ))
    assert_size_stride(primals_475, (80, ), (1, ))
    assert_size_stride(primals_476, (), ())
    assert_size_stride(primals_477, (480, ), (1, ))
    assert_size_stride(primals_478, (480, ), (1, ))
    assert_size_stride(primals_479, (), ())
    assert_size_stride(primals_480, (480, ), (1, ))
    assert_size_stride(primals_481, (480, ), (1, ))
    assert_size_stride(primals_482, (), ())
    assert_size_stride(primals_483, (80, ), (1, ))
    assert_size_stride(primals_484, (80, ), (1, ))
    assert_size_stride(primals_485, (), ())
    assert_size_stride(primals_486, (80, ), (1, ))
    assert_size_stride(primals_487, (80, ), (1, ))
    assert_size_stride(primals_488, (), ())
    assert_size_stride(primals_489, (480, ), (1, ))
    assert_size_stride(primals_490, (480, ), (1, ))
    assert_size_stride(primals_491, (), ())
    assert_size_stride(primals_492, (480, ), (1, ))
    assert_size_stride(primals_493, (480, ), (1, ))
    assert_size_stride(primals_494, (), ())
    assert_size_stride(primals_495, (80, ), (1, ))
    assert_size_stride(primals_496, (80, ), (1, ))
    assert_size_stride(primals_497, (), ())
    assert_size_stride(primals_498, (80, ), (1, ))
    assert_size_stride(primals_499, (80, ), (1, ))
    assert_size_stride(primals_500, (), ())
    assert_size_stride(primals_501, (480, ), (1, ))
    assert_size_stride(primals_502, (480, ), (1, ))
    assert_size_stride(primals_503, (), ())
    assert_size_stride(primals_504, (480, ), (1, ))
    assert_size_stride(primals_505, (480, ), (1, ))
    assert_size_stride(primals_506, (), ())
    assert_size_stride(primals_507, (80, ), (1, ))
    assert_size_stride(primals_508, (80, ), (1, ))
    assert_size_stride(primals_509, (), ())
    assert_size_stride(primals_510, (80, ), (1, ))
    assert_size_stride(primals_511, (80, ), (1, ))
    assert_size_stride(primals_512, (), ())
    assert_size_stride(primals_513, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_5.data_ptr()), c_void_p(primals_513.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_5
    del primals_513
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    buf3 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf6 = empty((16, ), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_1(c_void_p(buf2.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_7
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, primals_8, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 8, 112, 112), (100352, 1, 896, 8))
    buf9 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf12 = empty((8, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    buf20 = reinterpret_tensor(buf21, (8, 8, 112, 112), (200704, 1, 1792, 16), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_2(c_void_p(buf8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf20.data_ptr()))
    del primals_10
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf13, primals_11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf14, (8, 8, 112, 112), (100352, 1, 896, 8))
    buf15 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf16 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf18 = empty((8, ), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf21, (8, 8, 112, 112), (200704, 1, 1792, 16), 8)  # alias
    buf587 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_3(c_void_p(buf14.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf587.data_ptr()))
    del primals_13
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf22, (8, 8, 112, 112), (100352, 1, 896, 8))
    buf23 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf26 = empty((8, ), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_4(c_void_p(buf22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    del primals_16
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, primals_17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
    assert_size_stride(buf28, (8, 8, 112, 112), (100352, 1, 896, 8))
    buf29 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf30 = empty_strided((1, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    buf32 = empty((8, ), device='cpu', dtype=torch.float32)
    buf33 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_5(c_void_p(buf28.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del primals_19
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf33, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (8, 24, 112, 112), (301056, 1, 2688, 24))
    buf35 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf36 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf38 = empty((24, ), device='cpu', dtype=torch.float32)
    buf39 = empty_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.float32)
    buf47 = empty_strided((8, 48, 112, 112), (602112, 1, 5376, 48), device='cpu', dtype=torch.float32)
    buf46 = reinterpret_tensor(buf47, (8, 24, 112, 112), (602112, 1, 5376, 48), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_6(c_void_p(buf34.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf46.data_ptr()))
    del primals_22
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, primals_23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf40, (8, 24, 112, 112), (301056, 1, 2688, 24))
    buf41 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf44 = empty((24, ), device='cpu', dtype=torch.float32)
    buf45 = reinterpret_tensor(buf47, (8, 24, 112, 112), (602112, 1, 5376, 48), 24)  # alias
    buf586 = empty_strided((8, 24, 112, 112), (301056, 1, 2688, 24), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_7(c_void_p(buf40.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf586.data_ptr()))
    del primals_25
    # Source Nodes: [x_7], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf47, primals_26, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
    assert_size_stride(buf48, (8, 48, 56, 56), (150528, 1, 2688, 48))
    buf49 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    buf50 = empty_strided((1, 48, 1, 1), (48, 1, 48, 48), device='cpu', dtype=torch.float32)
    buf52 = empty((48, ), device='cpu', dtype=torch.float32)
    buf53 = empty_strided((8, 48, 56, 56), (150528, 1, 2688, 48), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_8(c_void_p(buf48.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    del primals_28
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf54 = extern_kernels.convolution(buf53, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (8, 12, 56, 56), (37632, 1, 672, 12))
    buf55 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf56 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf58 = empty((12, ), device='cpu', dtype=torch.float32)
    buf59 = empty_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_9(c_void_p(buf54.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()))
    del primals_31
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, primals_32, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
    assert_size_stride(buf60, (8, 12, 56, 56), (37632, 1, 672, 12))
    buf61 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf62 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf64 = empty((12, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_10(c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_0], Original ATen: [aten.convolution]
    buf65 = extern_kernels.convolution(buf33, primals_35, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
    assert_size_stride(buf65, (8, 16, 56, 56), (50176, 1, 896, 16))
    buf66 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf67 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf69 = empty((16, ), device='cpu', dtype=torch.float32)
    buf70 = empty_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_11(c_void_p(buf65.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_37
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___shortcut_2], Original ATen: [aten.convolution]
    buf71 = extern_kernels.convolution(buf70, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (8, 24, 56, 56), (75264, 1, 1344, 24))
    buf72 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf73 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf75 = empty((24, ), device='cpu', dtype=torch.float32)
    buf76 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_12(c_void_p(buf71.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    del primals_34
    del primals_40
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf77 = extern_kernels.convolution(buf76, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf77, (8, 36, 56, 56), (112896, 1, 2016, 36))
    buf78 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf81 = empty((36, ), device='cpu', dtype=torch.float32)
    buf82 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    buf90 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    buf89 = reinterpret_tensor(buf90, (8, 36, 56, 56), (225792, 1, 4032, 72), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_13(c_void_p(buf77.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf89.data_ptr()))
    del primals_43
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf83 = extern_kernels.convolution(buf82, primals_44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
    assert_size_stride(buf83, (8, 36, 56, 56), (112896, 1, 2016, 36))
    buf84 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf85 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf87 = empty((36, ), device='cpu', dtype=torch.float32)
    buf88 = reinterpret_tensor(buf90, (8, 36, 56, 56), (225792, 1, 4032, 72), 36)  # alias
    buf585 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_14(c_void_p(buf83.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf585.data_ptr()))
    del primals_46
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(buf90, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (8, 12, 56, 56), (37632, 1, 672, 12))
    buf92 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf93 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf95 = empty((12, ), device='cpu', dtype=torch.float32)
    buf96 = empty_strided((8, 12, 56, 56), (37632, 1, 672, 12), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_15(c_void_p(buf91.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    del primals_49
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf97 = extern_kernels.convolution(buf96, primals_50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=12, bias=None)
    assert_size_stride(buf97, (8, 12, 56, 56), (37632, 1, 672, 12))
    buf98 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf99 = empty_strided((1, 12, 1, 1), (12, 1, 12, 12), device='cpu', dtype=torch.float32)
    buf101 = empty((12, ), device='cpu', dtype=torch.float32)
    buf102 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_16(c_void_p(buf97.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del primals_52
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf103 = extern_kernels.convolution(buf102, primals_53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf103, (8, 36, 56, 56), (112896, 1, 2016, 36))
    buf104 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf105 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf107 = empty((36, ), device='cpu', dtype=torch.float32)
    buf108 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.float32)
    buf116 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    buf115 = reinterpret_tensor(buf116, (8, 36, 56, 56), (225792, 1, 4032, 72), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_17(c_void_p(buf103.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf115.data_ptr()))
    del primals_55
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf109 = extern_kernels.convolution(buf108, primals_56, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=36, bias=None)
    assert_size_stride(buf109, (8, 36, 56, 56), (112896, 1, 2016, 36))
    buf110 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf111 = empty_strided((1, 36, 1, 1), (36, 1, 36, 36), device='cpu', dtype=torch.float32)
    buf113 = empty((36, ), device='cpu', dtype=torch.float32)
    buf114 = reinterpret_tensor(buf116, (8, 36, 56, 56), (225792, 1, 4032, 72), 36)  # alias
    buf584 = empty_strided((8, 36, 56, 56), (112896, 1, 2016, 36), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_18(c_void_p(buf109.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf584.data_ptr()))
    del primals_58
    # Source Nodes: [x_15], Original ATen: [aten.convolution]
    buf117 = extern_kernels.convolution(buf116, primals_59, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
    assert_size_stride(buf117, (8, 72, 28, 28), (56448, 1, 2016, 72))
    buf118 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf119 = empty_strided((1, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf121 = empty((72, ), device='cpu', dtype=torch.float32)
    buf122 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    buf123 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cpu', dtype=torch.float32)
    buf124 = reinterpret_tensor(buf123, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf123  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_19(c_void_p(buf124.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del primals_61
    # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
    buf125 = extern_kernels.convolution(buf124, primals_62, primals_63, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf125, (8, 20, 1, 1), (20, 1, 20, 20))
    del primals_63
    buf126 = buf125; del buf125  # reuse
    cpp_fused_relu_20(c_void_p(buf126.data_ptr()))
    # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(buf126, primals_64, primals_65, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf127, (8, 72, 1, 1), (72, 1, 72, 72))
    del primals_65
    buf128 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    buf129 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_21(c_void_p(buf127.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf130 = extern_kernels.convolution(buf129, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf130, (8, 20, 28, 28), (15680, 1, 560, 20))
    buf131 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf132 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf134 = empty((20, ), device='cpu', dtype=torch.float32)
    buf135 = empty_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_22(c_void_p(buf130.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()))
    del primals_68
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(buf135, primals_69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
    assert_size_stride(buf136, (8, 20, 28, 28), (15680, 1, 560, 20))
    buf137 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf138 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf140 = empty((20, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_23(c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_0], Original ATen: [aten.convolution]
    buf141 = extern_kernels.convolution(buf102, primals_72, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
    assert_size_stride(buf141, (8, 24, 28, 28), (18816, 1, 672, 24))
    buf142 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf143 = empty_strided((1, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    buf145 = empty((24, ), device='cpu', dtype=torch.float32)
    buf146 = empty_strided((8, 24, 28, 28), (18816, 1, 672, 24), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_24(c_void_p(buf141.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    del primals_74
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___shortcut_2], Original ATen: [aten.convolution]
    buf147 = extern_kernels.convolution(buf146, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf147, (8, 40, 28, 28), (31360, 1, 1120, 40))
    buf148 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf149 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf151 = empty((40, ), device='cpu', dtype=torch.float32)
    buf152 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_25(c_void_p(buf147.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    del primals_71
    del primals_77
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf153 = extern_kernels.convolution(buf152, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf153, (8, 60, 28, 28), (47040, 1, 1680, 60))
    buf154 = empty_strided((1, 60, 1, 1), (60, 1, 60, 60), device='cpu', dtype=torch.float32)
    buf155 = empty_strided((1, 60, 1, 1), (60, 1, 60, 60), device='cpu', dtype=torch.float32)
    buf157 = empty((60, ), device='cpu', dtype=torch.float32)
    buf158 = empty_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cpu', dtype=torch.float32)
    buf166 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    buf165 = reinterpret_tensor(buf166, (8, 60, 28, 28), (94080, 1, 3360, 120), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_26(c_void_p(buf153.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf165.data_ptr()))
    del primals_80
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf159 = extern_kernels.convolution(buf158, primals_81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf159, (8, 60, 28, 28), (47040, 1, 1680, 60))
    buf160 = empty_strided((1, 60, 1, 1), (60, 1, 60, 60), device='cpu', dtype=torch.float32)
    buf161 = empty_strided((1, 60, 1, 1), (60, 1, 60, 60), device='cpu', dtype=torch.float32)
    buf163 = empty((60, ), device='cpu', dtype=torch.float32)
    buf164 = reinterpret_tensor(buf166, (8, 60, 28, 28), (94080, 1, 3360, 120), 60)  # alias
    buf582 = empty_strided((8, 60, 28, 28), (47040, 1, 1680, 60), device='cpu', dtype=torch.bool)
    buf167 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf168 = reinterpret_tensor(buf167, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf167  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_27(c_void_p(buf168.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf582.data_ptr()))
    del primals_83
    # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
    buf169 = extern_kernels.convolution(buf168, primals_84, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf169, (8, 32, 1, 1), (32, 1, 32, 32))
    del primals_85
    buf170 = buf169; del buf169  # reuse
    cpp_fused_relu_28(c_void_p(buf170.data_ptr()))
    # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
    buf171 = extern_kernels.convolution(buf170, primals_86, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf171, (8, 120, 1, 1), (120, 1, 120, 120))
    del primals_87
    buf172 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf173 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_29(c_void_p(buf171.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf174 = extern_kernels.convolution(buf173, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf174, (8, 20, 28, 28), (15680, 1, 560, 20))
    buf175 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf176 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf178 = empty((20, ), device='cpu', dtype=torch.float32)
    buf179 = empty_strided((8, 20, 28, 28), (15680, 1, 560, 20), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_30(c_void_p(buf174.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    del primals_90
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf180 = extern_kernels.convolution(buf179, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=20, bias=None)
    assert_size_stride(buf180, (8, 20, 28, 28), (15680, 1, 560, 20))
    buf181 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf182 = empty_strided((1, 20, 1, 1), (20, 1, 20, 20), device='cpu', dtype=torch.float32)
    buf184 = empty((20, ), device='cpu', dtype=torch.float32)
    buf185 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_31(c_void_p(buf180.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    del primals_93
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf186 = extern_kernels.convolution(buf185, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf186, (8, 120, 28, 28), (94080, 1, 3360, 120))
    buf187 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf188 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf190 = empty((120, ), device='cpu', dtype=torch.float32)
    buf191 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    buf199 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    buf198 = reinterpret_tensor(buf199, (8, 120, 28, 28), (188160, 1, 6720, 240), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_32(c_void_p(buf186.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf198.data_ptr()))
    del primals_96
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf192 = extern_kernels.convolution(buf191, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf192, (8, 120, 28, 28), (94080, 1, 3360, 120))
    buf193 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf194 = empty_strided((1, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    buf196 = empty((120, ), device='cpu', dtype=torch.float32)
    buf197 = reinterpret_tensor(buf199, (8, 120, 28, 28), (188160, 1, 6720, 240), 120)  # alias
    buf580 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_33(c_void_p(buf192.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf580.data_ptr()))
    del primals_99
    # Source Nodes: [x_25], Original ATen: [aten.convolution]
    buf200 = extern_kernels.convolution(buf199, primals_100, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf200, (8, 240, 14, 14), (47040, 1, 3360, 240))
    buf201 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf202 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf204 = empty((240, ), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_34(c_void_p(buf200.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del primals_102
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf206 = extern_kernels.convolution(buf205, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf206, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf207 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf208 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf210 = empty((40, ), device='cpu', dtype=torch.float32)
    buf211 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_35(c_void_p(buf206.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del primals_105
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf212 = extern_kernels.convolution(buf211, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf212, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf213 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf214 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf216 = empty((40, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_36(c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_0], Original ATen: [aten.convolution]
    buf217 = extern_kernels.convolution(buf185, primals_109, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf217, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf218 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf219 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf221 = empty((40, ), device='cpu', dtype=torch.float32)
    buf222 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_37(c_void_p(buf217.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del primals_111
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___shortcut_2], Original ATen: [aten.convolution]
    buf223 = extern_kernels.convolution(buf222, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf223, (8, 80, 14, 14), (15680, 1, 1120, 80))
    buf224 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf225 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf227 = empty((80, ), device='cpu', dtype=torch.float32)
    buf228 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_38(c_void_p(buf223.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del primals_108
    del primals_114
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf229 = extern_kernels.convolution(buf228, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf229, (8, 100, 14, 14), (19600, 1, 1400, 100))
    buf230 = empty_strided((1, 100, 1, 1), (100, 1, 100, 100), device='cpu', dtype=torch.float32)
    buf231 = empty_strided((1, 100, 1, 1), (100, 1, 100, 100), device='cpu', dtype=torch.float32)
    buf233 = empty((100, ), device='cpu', dtype=torch.float32)
    buf234 = empty_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cpu', dtype=torch.float32)
    buf242 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    buf241 = reinterpret_tensor(buf242, (8, 100, 14, 14), (39200, 1, 2800, 200), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_39(c_void_p(buf229.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf241.data_ptr()))
    del primals_117
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf235 = extern_kernels.convolution(buf234, primals_118, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=100, bias=None)
    assert_size_stride(buf235, (8, 100, 14, 14), (19600, 1, 1400, 100))
    buf236 = empty_strided((1, 100, 1, 1), (100, 1, 100, 100), device='cpu', dtype=torch.float32)
    buf237 = empty_strided((1, 100, 1, 1), (100, 1, 100, 100), device='cpu', dtype=torch.float32)
    buf239 = empty((100, ), device='cpu', dtype=torch.float32)
    buf240 = reinterpret_tensor(buf242, (8, 100, 14, 14), (39200, 1, 2800, 200), 100)  # alias
    buf579 = empty_strided((8, 100, 14, 14), (19600, 1, 1400, 100), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_40(c_void_p(buf235.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf579.data_ptr()))
    del primals_120
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf243 = extern_kernels.convolution(buf242, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf243, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf244 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf245 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf247 = empty((40, ), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_41(c_void_p(buf243.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    del primals_123
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf249 = extern_kernels.convolution(buf248, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf249, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf250 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf251 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf253 = empty((40, ), device='cpu', dtype=torch.float32)
    buf254 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_42(c_void_p(buf249.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del primals_126
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf255 = extern_kernels.convolution(buf254, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf255, (8, 92, 14, 14), (18032, 1, 1288, 92))
    buf256 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf257 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf259 = empty((92, ), device='cpu', dtype=torch.float32)
    buf260 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    buf268 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    buf267 = reinterpret_tensor(buf268, (8, 92, 14, 14), (36064, 1, 2576, 184), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_43(c_void_p(buf255.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf267.data_ptr()))
    del primals_129
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf261 = extern_kernels.convolution(buf260, primals_130, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
    assert_size_stride(buf261, (8, 92, 14, 14), (18032, 1, 1288, 92))
    buf262 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf263 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf265 = empty((92, ), device='cpu', dtype=torch.float32)
    buf266 = reinterpret_tensor(buf268, (8, 92, 14, 14), (36064, 1, 2576, 184), 92)  # alias
    buf578 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_44(c_void_p(buf261.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf578.data_ptr()))
    del primals_132
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf269 = extern_kernels.convolution(buf268, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf269, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf270 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf271 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf273 = empty((40, ), device='cpu', dtype=torch.float32)
    buf274 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_45(c_void_p(buf269.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del primals_135
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf275 = extern_kernels.convolution(buf274, primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf275, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf276 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf277 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf279 = empty((40, ), device='cpu', dtype=torch.float32)
    buf280 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_46(c_void_p(buf275.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf280.data_ptr()))
    del primals_138
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf281 = extern_kernels.convolution(buf280, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf281, (8, 92, 14, 14), (18032, 1, 1288, 92))
    buf282 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf283 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf285 = empty((92, ), device='cpu', dtype=torch.float32)
    buf286 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.float32)
    buf294 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    buf293 = reinterpret_tensor(buf294, (8, 92, 14, 14), (36064, 1, 2576, 184), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_47(c_void_p(buf281.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf293.data_ptr()))
    del primals_141
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf287 = extern_kernels.convolution(buf286, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=92, bias=None)
    assert_size_stride(buf287, (8, 92, 14, 14), (18032, 1, 1288, 92))
    buf288 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf289 = empty_strided((1, 92, 1, 1), (92, 1, 92, 92), device='cpu', dtype=torch.float32)
    buf291 = empty((92, ), device='cpu', dtype=torch.float32)
    buf292 = reinterpret_tensor(buf294, (8, 92, 14, 14), (36064, 1, 2576, 184), 92)  # alias
    buf577 = empty_strided((8, 92, 14, 14), (18032, 1, 1288, 92), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_48(c_void_p(buf287.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf577.data_ptr()))
    del primals_144
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf295 = extern_kernels.convolution(buf294, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf295, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf296 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf297 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf299 = empty((40, ), device='cpu', dtype=torch.float32)
    buf300 = empty_strided((8, 40, 14, 14), (7840, 1, 560, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_49(c_void_p(buf295.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del primals_147
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf301 = extern_kernels.convolution(buf300, primals_148, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=40, bias=None)
    assert_size_stride(buf301, (8, 40, 14, 14), (7840, 1, 560, 40))
    buf302 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf303 = empty_strided((1, 40, 1, 1), (40, 1, 40, 40), device='cpu', dtype=torch.float32)
    buf305 = empty((40, ), device='cpu', dtype=torch.float32)
    buf306 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_50(c_void_p(buf301.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del primals_150
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf307 = extern_kernels.convolution(buf306, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf307, (8, 240, 14, 14), (47040, 1, 3360, 240))
    buf308 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf309 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf311 = empty((240, ), device='cpu', dtype=torch.float32)
    buf312 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    buf320 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf319 = reinterpret_tensor(buf320, (8, 240, 14, 14), (94080, 1, 6720, 480), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_51(c_void_p(buf307.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf319.data_ptr()))
    del primals_153
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf313 = extern_kernels.convolution(buf312, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf313, (8, 240, 14, 14), (47040, 1, 3360, 240))
    buf314 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf315 = empty_strided((1, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    buf317 = empty((240, ), device='cpu', dtype=torch.float32)
    buf318 = reinterpret_tensor(buf320, (8, 240, 14, 14), (94080, 1, 6720, 480), 240)  # alias
    buf576 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.bool)
    buf321 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf322 = reinterpret_tensor(buf321, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf321  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_52(c_void_p(buf322.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf576.data_ptr()))
    del primals_156
    # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
    buf323 = extern_kernels.convolution(buf322, primals_157, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf323, (8, 120, 1, 1), (120, 1, 120, 120))
    del primals_158
    buf324 = buf323; del buf323  # reuse
    cpp_fused_relu_53(c_void_p(buf324.data_ptr()))
    # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
    buf325 = extern_kernels.convolution(buf324, primals_159, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf325, (8, 480, 1, 1), (480, 1, 480, 480))
    del primals_160
    buf326 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf327 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_54(c_void_p(buf325.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf328 = extern_kernels.convolution(buf327, primals_161, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf328, (8, 56, 14, 14), (10976, 1, 784, 56))
    buf329 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf330 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf332 = empty((56, ), device='cpu', dtype=torch.float32)
    buf333 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_55(c_void_p(buf328.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del primals_163
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf334 = extern_kernels.convolution(buf333, primals_164, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
    assert_size_stride(buf334, (8, 56, 14, 14), (10976, 1, 784, 56))
    buf335 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf336 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf338 = empty((56, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_56(c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_0], Original ATen: [aten.convolution]
    buf339 = extern_kernels.convolution(buf306, primals_167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf339, (8, 80, 14, 14), (15680, 1, 1120, 80))
    buf340 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf341 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf343 = empty((80, ), device='cpu', dtype=torch.float32)
    buf344 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_57(c_void_p(buf339.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del primals_169
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____3___shortcut_2], Original ATen: [aten.convolution]
    buf345 = extern_kernels.convolution(buf344, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf345, (8, 112, 14, 14), (21952, 1, 1568, 112))
    buf346 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf347 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf349 = empty((112, ), device='cpu', dtype=torch.float32)
    buf350 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_58(c_void_p(buf345.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(primals_165.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    del primals_166
    del primals_172
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf351 = extern_kernels.convolution(buf350, primals_173, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf351, (8, 336, 14, 14), (65856, 1, 4704, 336))
    buf352 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf353 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf355 = empty((336, ), device='cpu', dtype=torch.float32)
    buf356 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    buf364 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf363 = reinterpret_tensor(buf364, (8, 336, 14, 14), (131712, 1, 9408, 672), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_59(c_void_p(buf351.data_ptr()), c_void_p(primals_174.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf363.data_ptr()))
    del primals_175
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf357 = extern_kernels.convolution(buf356, primals_176, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
    assert_size_stride(buf357, (8, 336, 14, 14), (65856, 1, 4704, 336))
    buf358 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf359 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf361 = empty((336, ), device='cpu', dtype=torch.float32)
    buf362 = reinterpret_tensor(buf364, (8, 336, 14, 14), (131712, 1, 9408, 672), 336)  # alias
    buf574 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.bool)
    buf365 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf366 = reinterpret_tensor(buf365, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf365  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_60(c_void_p(buf366.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_177.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf574.data_ptr()))
    del primals_178
    # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
    buf367 = extern_kernels.convolution(buf366, primals_179, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf367, (8, 168, 1, 1), (168, 1, 168, 168))
    del primals_180
    buf368 = buf367; del buf367  # reuse
    cpp_fused_relu_61(c_void_p(buf368.data_ptr()))
    # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
    buf369 = extern_kernels.convolution(buf368, primals_181, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf369, (8, 672, 1, 1), (672, 1, 672, 672))
    del primals_182
    buf370 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf371 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_62(c_void_p(buf369.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf371.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf372 = extern_kernels.convolution(buf371, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf372, (8, 56, 14, 14), (10976, 1, 784, 56))
    buf373 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf374 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf376 = empty((56, ), device='cpu', dtype=torch.float32)
    buf377 = empty_strided((8, 56, 14, 14), (10976, 1, 784, 56), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_63(c_void_p(buf372.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    del primals_185
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____4___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf378 = extern_kernels.convolution(buf377, primals_186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=56, bias=None)
    assert_size_stride(buf378, (8, 56, 14, 14), (10976, 1, 784, 56))
    buf379 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf380 = empty_strided((1, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    buf382 = empty((56, ), device='cpu', dtype=torch.float32)
    buf383 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_64(c_void_p(buf378.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    del primals_188
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf384 = extern_kernels.convolution(buf383, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf384, (8, 336, 14, 14), (65856, 1, 4704, 336))
    buf385 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf386 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf388 = empty((336, ), device='cpu', dtype=torch.float32)
    buf389 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    buf397 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    buf396 = reinterpret_tensor(buf397, (8, 336, 14, 14), (131712, 1, 9408, 672), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_65(c_void_p(buf384.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf396.data_ptr()))
    del primals_191
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf390 = extern_kernels.convolution(buf389, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
    assert_size_stride(buf390, (8, 336, 14, 14), (65856, 1, 4704, 336))
    buf391 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf392 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cpu', dtype=torch.float32)
    buf394 = empty((336, ), device='cpu', dtype=torch.float32)
    buf395 = reinterpret_tensor(buf397, (8, 336, 14, 14), (131712, 1, 9408, 672), 336)  # alias
    buf572 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_66(c_void_p(buf390.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf572.data_ptr()))
    del primals_194
    # Source Nodes: [x_47], Original ATen: [aten.convolution]
    buf398 = extern_kernels.convolution(buf397, primals_195, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
    assert_size_stride(buf398, (8, 672, 7, 7), (32928, 1, 4704, 672))
    buf399 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf400 = empty_strided((1, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf402 = empty((672, ), device='cpu', dtype=torch.float32)
    buf403 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    buf404 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cpu', dtype=torch.float32)
    buf405 = reinterpret_tensor(buf404, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf404  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_67(c_void_p(buf405.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()))
    del primals_197
    # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
    buf406 = extern_kernels.convolution(buf405, primals_198, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf406, (8, 168, 1, 1), (168, 1, 168, 168))
    del primals_199
    buf407 = buf406; del buf406  # reuse
    cpp_fused_relu_68(c_void_p(buf407.data_ptr()))
    # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
    buf408 = extern_kernels.convolution(buf407, primals_200, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf408, (8, 672, 1, 1), (672, 1, 672, 672))
    del primals_201
    buf409 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    buf410 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_69(c_void_p(buf408.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf411 = extern_kernels.convolution(buf410, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf411, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf412 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf413 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf415 = empty((80, ), device='cpu', dtype=torch.float32)
    buf416 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_70(c_void_p(buf411.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_204.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()))
    del primals_204
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf417 = extern_kernels.convolution(buf416, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf417, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf418 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf419 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf421 = empty((80, ), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_71(c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf421.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_0], Original ATen: [aten.convolution]
    buf422 = extern_kernels.convolution(buf383, primals_208, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf422, (8, 112, 7, 7), (5488, 1, 784, 112))
    buf423 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf424 = empty_strided((1, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    buf426 = empty((112, ), device='cpu', dtype=torch.float32)
    buf427 = empty_strided((8, 112, 7, 7), (5488, 1, 784, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_72(c_void_p(buf422.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    del primals_210
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___shortcut_2], Original ATen: [aten.convolution]
    buf428 = extern_kernels.convolution(buf427, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf428, (8, 160, 7, 7), (7840, 1, 1120, 160))
    buf429 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf430 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cpu', dtype=torch.float32)
    buf432 = empty((160, ), device='cpu', dtype=torch.float32)
    buf433 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_73(c_void_p(buf428.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()))
    del primals_207
    del primals_213
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf434 = extern_kernels.convolution(buf433, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf434, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf435 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf436 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf438 = empty((480, ), device='cpu', dtype=torch.float32)
    buf439 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    buf447 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf446 = reinterpret_tensor(buf447, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_74(c_void_p(buf434.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf446.data_ptr()))
    del primals_216
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf440 = extern_kernels.convolution(buf439, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf440, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf441 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf442 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf444 = empty((480, ), device='cpu', dtype=torch.float32)
    buf445 = reinterpret_tensor(buf447, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
    buf570 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_75(c_void_p(buf440.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf570.data_ptr()))
    del primals_219
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf448 = extern_kernels.convolution(buf447, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf448, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf449 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf450 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf452 = empty((80, ), device='cpu', dtype=torch.float32)
    buf453 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_76(c_void_p(buf448.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    del primals_222
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf454 = extern_kernels.convolution(buf453, primals_223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf454, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf455 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf456 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf458 = empty((80, ), device='cpu', dtype=torch.float32)
    buf459 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_77(c_void_p(buf454.data_ptr()), c_void_p(buf453.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()))
    del primals_225
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf460 = extern_kernels.convolution(buf459, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf460, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf461 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf462 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf464 = empty((480, ), device='cpu', dtype=torch.float32)
    buf465 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    buf473 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf472 = reinterpret_tensor(buf473, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_78(c_void_p(buf460.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf472.data_ptr()))
    del primals_228
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf466 = extern_kernels.convolution(buf465, primals_229, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf466, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf467 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf468 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf470 = empty((480, ), device='cpu', dtype=torch.float32)
    buf471 = reinterpret_tensor(buf473, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
    buf569 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    buf474 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cpu', dtype=torch.float32)
    buf475 = reinterpret_tensor(buf474, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf474  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_79(c_void_p(buf475.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf569.data_ptr()))
    del primals_231
    # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
    buf476 = extern_kernels.convolution(buf475, primals_232, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf476, (8, 240, 1, 1), (240, 1, 240, 240))
    del primals_233
    buf477 = buf476; del buf476  # reuse
    cpp_fused_relu_80(c_void_p(buf477.data_ptr()))
    # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
    buf478 = extern_kernels.convolution(buf477, primals_234, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf478, (8, 960, 1, 1), (960, 1, 960, 960))
    del primals_235
    buf479 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf480 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_mul_81(c_void_p(buf478.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf481 = extern_kernels.convolution(buf480, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf481, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf482 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf483 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf485 = empty((80, ), device='cpu', dtype=torch.float32)
    buf486 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_82(c_void_p(buf481.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    del primals_238
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____1___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf487 = extern_kernels.convolution(buf486, primals_239, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf487, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf488 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf489 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf491 = empty((80, ), device='cpu', dtype=torch.float32)
    buf492 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_83(c_void_p(buf487.data_ptr()), c_void_p(buf486.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()))
    del primals_241
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf493 = extern_kernels.convolution(buf492, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf493, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf494 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf495 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf497 = empty((480, ), device='cpu', dtype=torch.float32)
    buf498 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    buf506 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf505 = reinterpret_tensor(buf506, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_84(c_void_p(buf493.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf498.data_ptr()), c_void_p(buf505.data_ptr()))
    del primals_244
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf499 = extern_kernels.convolution(buf498, primals_245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf499, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf500 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf501 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf503 = empty((480, ), device='cpu', dtype=torch.float32)
    buf504 = reinterpret_tensor(buf506, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
    buf567 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    cpp_fused__native_batch_norm_legit_functional_relu_threshold_backward_85(c_void_p(buf499.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf567.data_ptr()))
    del primals_247
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf507 = extern_kernels.convolution(buf506, primals_248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf507, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf508 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf509 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf511 = empty((80, ), device='cpu', dtype=torch.float32)
    buf512 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_86(c_void_p(buf507.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf511.data_ptr()), c_void_p(buf512.data_ptr()))
    del primals_250
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____2___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf513 = extern_kernels.convolution(buf512, primals_251, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf513, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf514 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf515 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf517 = empty((80, ), device='cpu', dtype=torch.float32)
    buf518 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_87(c_void_p(buf513.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(buf518.data_ptr()))
    del primals_253
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_primary_conv_0], Original ATen: [aten.convolution]
    buf519 = extern_kernels.convolution(buf518, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf519, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf520 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf521 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf523 = empty((480, ), device='cpu', dtype=torch.float32)
    buf524 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.float32)
    buf532 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf531 = reinterpret_tensor(buf532, (8, 480, 7, 7), (47040, 1, 6720, 960), 0)  # alias
    cpp_fused__native_batch_norm_legit_functional_cat_relu_88(c_void_p(buf519.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf523.data_ptr()), c_void_p(buf524.data_ptr()), c_void_p(buf531.data_ptr()))
    del primals_256
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost1_cheap_operation_0], Original ATen: [aten.convolution]
    buf525 = extern_kernels.convolution(buf524, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
    assert_size_stride(buf525, (8, 480, 7, 7), (23520, 1, 3360, 480))
    buf526 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf527 = empty_strided((1, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf529 = empty((480, ), device='cpu', dtype=torch.float32)
    buf530 = reinterpret_tensor(buf532, (8, 480, 7, 7), (47040, 1, 6720, 960), 480)  # alias
    buf566 = empty_strided((8, 480, 7, 7), (23520, 1, 3360, 480), device='cpu', dtype=torch.bool)
    buf533 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cpu', dtype=torch.float32)
    buf534 = reinterpret_tensor(buf533, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf533  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_89(c_void_p(buf534.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()), c_void_p(buf566.data_ptr()))
    del primals_259
    # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
    buf535 = extern_kernels.convolution(buf534, primals_260, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf535, (8, 240, 1, 1), (240, 1, 240, 240))
    del primals_261
    buf536 = buf535; del buf535  # reuse
    cpp_fused_relu_90(c_void_p(buf536.data_ptr()))
    # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
    buf537 = extern_kernels.convolution(buf536, primals_262, primals_263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf537, (8, 960, 1, 1), (960, 1, 960, 960))
    del primals_263
    buf538 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf565 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    buf539 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused_hardsigmoid_hardsigmoid_backward_mul_91(c_void_p(buf537.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(buf539.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_primary_conv_0], Original ATen: [aten.convolution]
    buf540 = extern_kernels.convolution(buf539, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf540, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf541 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf542 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf544 = empty((80, ), device='cpu', dtype=torch.float32)
    buf545 = empty_strided((8, 80, 7, 7), (3920, 1, 560, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_92(c_void_p(buf540.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf542.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()))
    del primals_266
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____3___ghost2_cheap_operation_0], Original ATen: [aten.convolution]
    buf546 = extern_kernels.convolution(buf545, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=80, bias=None)
    assert_size_stride(buf546, (8, 80, 7, 7), (3920, 1, 560, 80))
    buf547 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf548 = empty_strided((1, 80, 1, 1), (80, 1, 80, 80), device='cpu', dtype=torch.float32)
    buf550 = empty((80, ), device='cpu', dtype=torch.float32)
    buf551 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_cat_93(c_void_p(buf546.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf551.data_ptr()))
    del primals_269
    # Source Nodes: [x_66], Original ATen: [aten.convolution]
    buf552 = extern_kernels.convolution(buf551, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf552, (8, 960, 7, 7), (47040, 1, 6720, 960))
    buf553 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf554 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    buf556 = empty((960, ), device='cpu', dtype=torch.float32)
    buf557 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    buf558 = reinterpret_tensor(buf537, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf537  # reuse
    buf559 = reinterpret_tensor(buf558, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf558  # reuse
    cpp_fused__native_batch_norm_legit_functional_mean_relu_94(c_void_p(buf559.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(buf557.data_ptr()))
    del buf554
    del primals_2
    del primals_275
    # Source Nodes: [x_76], Original ATen: [aten.convolution]
    buf560 = extern_kernels.convolution(buf559, primals_271, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf560, (8, 1280, 1, 1), (1280, 1, 1280, 1280))
    del primals_272
    buf561 = empty((8, 1280), device='cpu', dtype=torch.float32)
    buf563 = empty_strided((8, 1280, 1, 1), (1280, 1, 1280, 1280), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_view_95(c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf563.data_ptr()))
    del buf560
    buf562 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_4, buf561, reinterpret_tensor(primals_3, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf562)
    del primals_4
    buf564 = empty_strided((8, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.bool)
    buf568 = empty_strided((8, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    buf571 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    buf573 = empty_strided((8, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    buf575 = empty_strided((8, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.bool)
    buf581 = empty_strided((8, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    buf583 = empty_strided((8, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.bool)
    buf600 = reinterpret_tensor(buf4, (16, ), (1, ), 0); del buf4  # reuse
    buf608 = reinterpret_tensor(buf10, (8, ), (1, ), 0); del buf10  # reuse
    buf616 = reinterpret_tensor(buf16, (8, ), (1, ), 0); del buf16  # reuse
    buf624 = reinterpret_tensor(buf24, (8, ), (1, ), 0); del buf24  # reuse
    buf632 = reinterpret_tensor(buf30, (8, ), (1, ), 0); del buf30  # reuse
    buf640 = reinterpret_tensor(buf36, (24, ), (1, ), 0); del buf36  # reuse
    buf648 = reinterpret_tensor(buf42, (24, ), (1, ), 0); del buf42  # reuse
    buf656 = reinterpret_tensor(buf50, (48, ), (1, ), 0); del buf50  # reuse
    buf664 = reinterpret_tensor(buf56, (12, ), (1, ), 0); del buf56  # reuse
    buf672 = reinterpret_tensor(buf62, (12, ), (1, ), 0); del buf62  # reuse
    buf680 = reinterpret_tensor(buf67, (16, ), (1, ), 0); del buf67  # reuse
    buf688 = reinterpret_tensor(buf73, (24, ), (1, ), 0); del buf73  # reuse
    buf696 = reinterpret_tensor(buf79, (36, ), (1, ), 0); del buf79  # reuse
    buf704 = reinterpret_tensor(buf85, (36, ), (1, ), 0); del buf85  # reuse
    buf712 = reinterpret_tensor(buf93, (12, ), (1, ), 0); del buf93  # reuse
    buf720 = reinterpret_tensor(buf99, (12, ), (1, ), 0); del buf99  # reuse
    buf728 = reinterpret_tensor(buf105, (36, ), (1, ), 0); del buf105  # reuse
    buf736 = reinterpret_tensor(buf111, (36, ), (1, ), 0); del buf111  # reuse
    buf744 = reinterpret_tensor(buf119, (72, ), (1, ), 0); del buf119  # reuse
    buf752 = reinterpret_tensor(buf132, (20, ), (1, ), 0); del buf132  # reuse
    buf760 = reinterpret_tensor(buf138, (20, ), (1, ), 0); del buf138  # reuse
    buf768 = reinterpret_tensor(buf143, (24, ), (1, ), 0); del buf143  # reuse
    buf776 = reinterpret_tensor(buf149, (40, ), (1, ), 0); del buf149  # reuse
    buf784 = reinterpret_tensor(buf155, (60, ), (1, ), 0); del buf155  # reuse
    buf792 = reinterpret_tensor(buf161, (60, ), (1, ), 0); del buf161  # reuse
    buf800 = reinterpret_tensor(buf176, (20, ), (1, ), 0); del buf176  # reuse
    buf808 = reinterpret_tensor(buf182, (20, ), (1, ), 0); del buf182  # reuse
    buf816 = reinterpret_tensor(buf188, (120, ), (1, ), 0); del buf188  # reuse
    buf824 = reinterpret_tensor(buf194, (120, ), (1, ), 0); del buf194  # reuse
    buf832 = reinterpret_tensor(buf202, (240, ), (1, ), 0); del buf202  # reuse
    buf840 = reinterpret_tensor(buf208, (40, ), (1, ), 0); del buf208  # reuse
    buf848 = reinterpret_tensor(buf214, (40, ), (1, ), 0); del buf214  # reuse
    buf856 = reinterpret_tensor(buf219, (40, ), (1, ), 0); del buf219  # reuse
    buf864 = reinterpret_tensor(buf225, (80, ), (1, ), 0); del buf225  # reuse
    buf872 = reinterpret_tensor(buf231, (100, ), (1, ), 0); del buf231  # reuse
    buf880 = reinterpret_tensor(buf237, (100, ), (1, ), 0); del buf237  # reuse
    buf888 = reinterpret_tensor(buf245, (40, ), (1, ), 0); del buf245  # reuse
    buf896 = reinterpret_tensor(buf251, (40, ), (1, ), 0); del buf251  # reuse
    buf904 = reinterpret_tensor(buf257, (92, ), (1, ), 0); del buf257  # reuse
    buf912 = reinterpret_tensor(buf263, (92, ), (1, ), 0); del buf263  # reuse
    buf920 = reinterpret_tensor(buf271, (40, ), (1, ), 0); del buf271  # reuse
    buf928 = reinterpret_tensor(buf277, (40, ), (1, ), 0); del buf277  # reuse
    buf936 = reinterpret_tensor(buf283, (92, ), (1, ), 0); del buf283  # reuse
    buf944 = reinterpret_tensor(buf289, (92, ), (1, ), 0); del buf289  # reuse
    buf952 = reinterpret_tensor(buf297, (40, ), (1, ), 0); del buf297  # reuse
    buf960 = reinterpret_tensor(buf303, (40, ), (1, ), 0); del buf303  # reuse
    buf968 = reinterpret_tensor(buf309, (240, ), (1, ), 0); del buf309  # reuse
    buf976 = reinterpret_tensor(buf315, (240, ), (1, ), 0); del buf315  # reuse
    buf984 = reinterpret_tensor(buf330, (56, ), (1, ), 0); del buf330  # reuse
    buf992 = reinterpret_tensor(buf336, (56, ), (1, ), 0); del buf336  # reuse
    buf1000 = reinterpret_tensor(buf341, (80, ), (1, ), 0); del buf341  # reuse
    buf1008 = reinterpret_tensor(buf347, (112, ), (1, ), 0); del buf347  # reuse
    buf1016 = reinterpret_tensor(buf353, (336, ), (1, ), 0); del buf353  # reuse
    buf1024 = reinterpret_tensor(buf359, (336, ), (1, ), 0); del buf359  # reuse
    buf1032 = reinterpret_tensor(buf374, (56, ), (1, ), 0); del buf374  # reuse
    buf1040 = reinterpret_tensor(buf380, (56, ), (1, ), 0); del buf380  # reuse
    buf1048 = reinterpret_tensor(buf386, (336, ), (1, ), 0); del buf386  # reuse
    buf1056 = reinterpret_tensor(buf392, (336, ), (1, ), 0); del buf392  # reuse
    buf1064 = reinterpret_tensor(buf400, (672, ), (1, ), 0); del buf400  # reuse
    buf1072 = reinterpret_tensor(buf413, (80, ), (1, ), 0); del buf413  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_hardsigmoid_backward_threshold_backward_96(c_void_p(buf600.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(buf640.data_ptr()), c_void_p(buf648.data_ptr()), c_void_p(buf656.data_ptr()), c_void_p(buf664.data_ptr()), c_void_p(buf672.data_ptr()), c_void_p(buf680.data_ptr()), c_void_p(buf688.data_ptr()), c_void_p(buf696.data_ptr()), c_void_p(buf704.data_ptr()), c_void_p(buf712.data_ptr()), c_void_p(buf720.data_ptr()), c_void_p(buf728.data_ptr()), c_void_p(buf736.data_ptr()), c_void_p(buf744.data_ptr()), c_void_p(buf752.data_ptr()), c_void_p(buf760.data_ptr()), c_void_p(buf768.data_ptr()), c_void_p(buf776.data_ptr()), c_void_p(buf784.data_ptr()), c_void_p(buf792.data_ptr()), c_void_p(buf800.data_ptr()), c_void_p(buf808.data_ptr()), c_void_p(buf816.data_ptr()), c_void_p(buf824.data_ptr()), c_void_p(buf832.data_ptr()), c_void_p(buf840.data_ptr()), c_void_p(buf848.data_ptr()), c_void_p(buf856.data_ptr()), c_void_p(buf864.data_ptr()), c_void_p(buf872.data_ptr()), c_void_p(buf880.data_ptr()), c_void_p(buf888.data_ptr()), c_void_p(buf896.data_ptr()), c_void_p(buf904.data_ptr()), c_void_p(buf912.data_ptr()), c_void_p(buf920.data_ptr()), c_void_p(buf928.data_ptr()), c_void_p(buf936.data_ptr()), c_void_p(buf944.data_ptr()), c_void_p(buf952.data_ptr()), c_void_p(buf960.data_ptr()), c_void_p(buf968.data_ptr()), c_void_p(buf976.data_ptr()), c_void_p(buf984.data_ptr()), c_void_p(buf992.data_ptr()), c_void_p(buf1000.data_ptr()), c_void_p(buf1008.data_ptr()), c_void_p(buf1016.data_ptr()), c_void_p(buf1024.data_ptr()), c_void_p(buf1032.data_ptr()), c_void_p(buf1040.data_ptr()), c_void_p(buf1048.data_ptr()), c_void_p(buf1056.data_ptr()), c_void_p(buf1064.data_ptr()), c_void_p(buf1072.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(primals_414.data_ptr()), c_void_p(primals_415.data_ptr()), c_void_p(primals_416.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_417.data_ptr()), c_void_p(primals_418.data_ptr()), c_void_p(primals_419.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_420.data_ptr()), c_void_p(primals_421.data_ptr()), c_void_p(primals_422.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(primals_423.data_ptr()), c_void_p(primals_424.data_ptr()), c_void_p(primals_425.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(primals_426.data_ptr()), c_void_p(primals_427.data_ptr()), c_void_p(primals_428.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(primals_429.data_ptr()), c_void_p(primals_430.data_ptr()), c_void_p(primals_431.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(primals_432.data_ptr()), c_void_p(primals_433.data_ptr()), c_void_p(primals_434.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(primals_435.data_ptr()), c_void_p(primals_436.data_ptr()), c_void_p(primals_437.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(primals_438.data_ptr()), c_void_p(primals_439.data_ptr()), c_void_p(primals_440.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(primals_441.data_ptr()), c_void_p(primals_442.data_ptr()), c_void_p(primals_443.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(primals_444.data_ptr()), c_void_p(primals_445.data_ptr()), c_void_p(primals_446.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(primals_447.data_ptr()), c_void_p(primals_448.data_ptr()), c_void_p(primals_449.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(primals_450.data_ptr()), c_void_p(primals_451.data_ptr()), c_void_p(primals_452.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(primals_453.data_ptr()), c_void_p(primals_454.data_ptr()), c_void_p(primals_455.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(primals_456.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(buf571.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(primals_414.data_ptr()), c_void_p(primals_415.data_ptr()), c_void_p(primals_416.data_ptr()), c_void_p(primals_417.data_ptr()), c_void_p(primals_418.data_ptr()), c_void_p(primals_419.data_ptr()), c_void_p(primals_420.data_ptr()), c_void_p(primals_421.data_ptr()), c_void_p(primals_422.data_ptr()), c_void_p(primals_423.data_ptr()), c_void_p(primals_424.data_ptr()), c_void_p(primals_425.data_ptr()), c_void_p(primals_426.data_ptr()), c_void_p(primals_427.data_ptr()), c_void_p(primals_428.data_ptr()), c_void_p(primals_429.data_ptr()), c_void_p(primals_430.data_ptr()), c_void_p(primals_431.data_ptr()), c_void_p(primals_432.data_ptr()), c_void_p(primals_433.data_ptr()), c_void_p(primals_434.data_ptr()), c_void_p(primals_435.data_ptr()), c_void_p(primals_436.data_ptr()), c_void_p(primals_437.data_ptr()), c_void_p(primals_438.data_ptr()), c_void_p(primals_439.data_ptr()), c_void_p(primals_440.data_ptr()), c_void_p(primals_441.data_ptr()), c_void_p(primals_442.data_ptr()), c_void_p(primals_443.data_ptr()), c_void_p(primals_444.data_ptr()), c_void_p(primals_445.data_ptr()), c_void_p(primals_446.data_ptr()), c_void_p(primals_447.data_ptr()), c_void_p(primals_448.data_ptr()), c_void_p(primals_449.data_ptr()), c_void_p(primals_450.data_ptr()), c_void_p(primals_451.data_ptr()), c_void_p(primals_452.data_ptr()), c_void_p(primals_453.data_ptr()), c_void_p(primals_454.data_ptr()), c_void_p(primals_455.data_ptr()), c_void_p(primals_456.data_ptr()))
    del buf1000
    del buf1008
    del buf1016
    del buf1024
    del buf1032
    del buf1040
    del buf1048
    del buf1056
    del buf1064
    del buf1072
    del buf127
    del buf171
    del buf325
    del buf369
    del buf408
    del buf478
    del buf557
    del buf600
    del buf608
    del buf616
    del buf624
    del buf632
    del buf640
    del buf648
    del buf656
    del buf664
    del buf672
    del buf680
    del buf688
    del buf696
    del buf704
    del buf712
    del buf720
    del buf728
    del buf736
    del buf744
    del buf752
    del buf760
    del buf768
    del buf776
    del buf784
    del buf792
    del buf800
    del buf808
    del buf816
    del buf824
    del buf832
    del buf840
    del buf848
    del buf856
    del buf864
    del buf872
    del buf880
    del buf888
    del buf896
    del buf904
    del buf912
    del buf920
    del buf928
    del buf936
    del buf944
    del buf952
    del buf960
    del buf968
    del buf976
    del buf984
    del buf992
    del primals_273
    del primals_274
    del primals_276
    del primals_277
    del primals_278
    del primals_279
    del primals_280
    del primals_281
    del primals_282
    del primals_283
    del primals_284
    del primals_285
    del primals_286
    del primals_287
    del primals_288
    del primals_289
    del primals_290
    del primals_291
    del primals_292
    del primals_293
    del primals_294
    del primals_295
    del primals_296
    del primals_297
    del primals_298
    del primals_299
    del primals_300
    del primals_301
    del primals_302
    del primals_303
    del primals_304
    del primals_305
    del primals_306
    del primals_307
    del primals_308
    del primals_309
    del primals_310
    del primals_311
    del primals_312
    del primals_313
    del primals_314
    del primals_315
    del primals_316
    del primals_317
    del primals_318
    del primals_319
    del primals_320
    del primals_321
    del primals_322
    del primals_323
    del primals_324
    del primals_325
    del primals_326
    del primals_327
    del primals_328
    del primals_329
    del primals_330
    del primals_331
    del primals_332
    del primals_333
    del primals_334
    del primals_335
    del primals_336
    del primals_337
    del primals_338
    del primals_339
    del primals_340
    del primals_341
    del primals_342
    del primals_343
    del primals_344
    del primals_345
    del primals_346
    del primals_347
    del primals_348
    del primals_349
    del primals_350
    del primals_351
    del primals_352
    del primals_353
    del primals_354
    del primals_355
    del primals_356
    del primals_357
    del primals_358
    del primals_359
    del primals_360
    del primals_361
    del primals_362
    del primals_363
    del primals_364
    del primals_365
    del primals_366
    del primals_367
    del primals_368
    del primals_369
    del primals_370
    del primals_371
    del primals_372
    del primals_373
    del primals_374
    del primals_375
    del primals_376
    del primals_377
    del primals_378
    del primals_379
    del primals_380
    del primals_381
    del primals_382
    del primals_383
    del primals_384
    del primals_385
    del primals_386
    del primals_387
    del primals_388
    del primals_389
    del primals_390
    del primals_391
    del primals_392
    del primals_393
    del primals_394
    del primals_395
    del primals_396
    del primals_397
    del primals_398
    del primals_399
    del primals_400
    del primals_401
    del primals_402
    del primals_403
    del primals_404
    del primals_405
    del primals_406
    del primals_407
    del primals_408
    del primals_409
    del primals_410
    del primals_411
    del primals_412
    del primals_413
    del primals_414
    del primals_415
    del primals_416
    del primals_417
    del primals_418
    del primals_419
    del primals_420
    del primals_421
    del primals_422
    del primals_423
    del primals_424
    del primals_425
    del primals_426
    del primals_427
    del primals_428
    del primals_429
    del primals_430
    del primals_431
    del primals_432
    del primals_433
    del primals_434
    del primals_435
    del primals_436
    del primals_437
    del primals_438
    del primals_439
    del primals_440
    del primals_441
    del primals_442
    del primals_443
    del primals_444
    del primals_445
    del primals_446
    del primals_447
    del primals_448
    del primals_449
    del primals_450
    del primals_451
    del primals_452
    del primals_453
    del primals_454
    del primals_455
    del primals_456
    buf1080 = reinterpret_tensor(buf419, (80, ), (1, ), 0); del buf419  # reuse
    buf1088 = reinterpret_tensor(buf424, (112, ), (1, ), 0); del buf424  # reuse
    buf1096 = reinterpret_tensor(buf430, (160, ), (1, ), 0); del buf430  # reuse
    buf1104 = reinterpret_tensor(buf436, (480, ), (1, ), 0); del buf436  # reuse
    buf1112 = reinterpret_tensor(buf442, (480, ), (1, ), 0); del buf442  # reuse
    buf1120 = reinterpret_tensor(buf450, (80, ), (1, ), 0); del buf450  # reuse
    buf1128 = reinterpret_tensor(buf456, (80, ), (1, ), 0); del buf456  # reuse
    buf1136 = reinterpret_tensor(buf462, (480, ), (1, ), 0); del buf462  # reuse
    buf1144 = reinterpret_tensor(buf468, (480, ), (1, ), 0); del buf468  # reuse
    buf1152 = reinterpret_tensor(buf483, (80, ), (1, ), 0); del buf483  # reuse
    buf1160 = reinterpret_tensor(buf489, (80, ), (1, ), 0); del buf489  # reuse
    buf1168 = reinterpret_tensor(buf495, (480, ), (1, ), 0); del buf495  # reuse
    buf1176 = reinterpret_tensor(buf501, (480, ), (1, ), 0); del buf501  # reuse
    buf1184 = reinterpret_tensor(buf509, (80, ), (1, ), 0); del buf509  # reuse
    buf1192 = reinterpret_tensor(buf515, (80, ), (1, ), 0); del buf515  # reuse
    buf1200 = reinterpret_tensor(buf521, (480, ), (1, ), 0); del buf521  # reuse
    buf1208 = reinterpret_tensor(buf527, (480, ), (1, ), 0); del buf527  # reuse
    buf1216 = reinterpret_tensor(buf542, (80, ), (1, ), 0); del buf542  # reuse
    buf1224 = reinterpret_tensor(buf548, (80, ), (1, ), 0); del buf548  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_97(c_void_p(buf1080.data_ptr()), c_void_p(buf1088.data_ptr()), c_void_p(buf1096.data_ptr()), c_void_p(buf1104.data_ptr()), c_void_p(buf1112.data_ptr()), c_void_p(buf1120.data_ptr()), c_void_p(buf1128.data_ptr()), c_void_p(buf1136.data_ptr()), c_void_p(buf1144.data_ptr()), c_void_p(buf1152.data_ptr()), c_void_p(buf1160.data_ptr()), c_void_p(buf1168.data_ptr()), c_void_p(buf1176.data_ptr()), c_void_p(buf1184.data_ptr()), c_void_p(buf1192.data_ptr()), c_void_p(buf1200.data_ptr()), c_void_p(buf1208.data_ptr()), c_void_p(buf1216.data_ptr()), c_void_p(buf1224.data_ptr()), c_void_p(primals_457.data_ptr()), c_void_p(primals_458.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(primals_459.data_ptr()), c_void_p(primals_460.data_ptr()), c_void_p(primals_461.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(primals_462.data_ptr()), c_void_p(primals_463.data_ptr()), c_void_p(primals_464.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(primals_465.data_ptr()), c_void_p(primals_466.data_ptr()), c_void_p(primals_467.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(primals_468.data_ptr()), c_void_p(primals_469.data_ptr()), c_void_p(primals_470.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(primals_471.data_ptr()), c_void_p(primals_472.data_ptr()), c_void_p(primals_473.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(primals_474.data_ptr()), c_void_p(primals_475.data_ptr()), c_void_p(primals_476.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(primals_477.data_ptr()), c_void_p(primals_478.data_ptr()), c_void_p(primals_479.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(primals_480.data_ptr()), c_void_p(primals_481.data_ptr()), c_void_p(primals_482.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(primals_483.data_ptr()), c_void_p(primals_484.data_ptr()), c_void_p(primals_485.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(primals_486.data_ptr()), c_void_p(primals_487.data_ptr()), c_void_p(primals_488.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(primals_489.data_ptr()), c_void_p(primals_490.data_ptr()), c_void_p(primals_491.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(primals_492.data_ptr()), c_void_p(primals_493.data_ptr()), c_void_p(primals_494.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(primals_495.data_ptr()), c_void_p(primals_496.data_ptr()), c_void_p(primals_497.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(primals_498.data_ptr()), c_void_p(primals_499.data_ptr()), c_void_p(primals_500.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(primals_501.data_ptr()), c_void_p(primals_502.data_ptr()), c_void_p(primals_503.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(primals_504.data_ptr()), c_void_p(primals_505.data_ptr()), c_void_p(primals_506.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(primals_507.data_ptr()), c_void_p(primals_508.data_ptr()), c_void_p(primals_509.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(primals_510.data_ptr()), c_void_p(primals_511.data_ptr()), c_void_p(primals_512.data_ptr()), c_void_p(primals_457.data_ptr()), c_void_p(primals_458.data_ptr()), c_void_p(primals_459.data_ptr()), c_void_p(primals_460.data_ptr()), c_void_p(primals_461.data_ptr()), c_void_p(primals_462.data_ptr()), c_void_p(primals_463.data_ptr()), c_void_p(primals_464.data_ptr()), c_void_p(primals_465.data_ptr()), c_void_p(primals_466.data_ptr()), c_void_p(primals_467.data_ptr()), c_void_p(primals_468.data_ptr()), c_void_p(primals_469.data_ptr()), c_void_p(primals_470.data_ptr()), c_void_p(primals_471.data_ptr()), c_void_p(primals_472.data_ptr()), c_void_p(primals_473.data_ptr()), c_void_p(primals_474.data_ptr()), c_void_p(primals_475.data_ptr()), c_void_p(primals_476.data_ptr()), c_void_p(primals_477.data_ptr()), c_void_p(primals_478.data_ptr()), c_void_p(primals_479.data_ptr()), c_void_p(primals_480.data_ptr()), c_void_p(primals_481.data_ptr()), c_void_p(primals_482.data_ptr()), c_void_p(primals_483.data_ptr()), c_void_p(primals_484.data_ptr()), c_void_p(primals_485.data_ptr()), c_void_p(primals_486.data_ptr()), c_void_p(primals_487.data_ptr()), c_void_p(primals_488.data_ptr()), c_void_p(primals_489.data_ptr()), c_void_p(primals_490.data_ptr()), c_void_p(primals_491.data_ptr()), c_void_p(primals_492.data_ptr()), c_void_p(primals_493.data_ptr()), c_void_p(primals_494.data_ptr()), c_void_p(primals_495.data_ptr()), c_void_p(primals_496.data_ptr()), c_void_p(primals_497.data_ptr()), c_void_p(primals_498.data_ptr()), c_void_p(primals_499.data_ptr()), c_void_p(primals_500.data_ptr()), c_void_p(primals_501.data_ptr()), c_void_p(primals_502.data_ptr()), c_void_p(primals_503.data_ptr()), c_void_p(primals_504.data_ptr()), c_void_p(primals_505.data_ptr()), c_void_p(primals_506.data_ptr()), c_void_p(primals_507.data_ptr()), c_void_p(primals_508.data_ptr()), c_void_p(primals_509.data_ptr()), c_void_p(primals_510.data_ptr()), c_void_p(primals_511.data_ptr()), c_void_p(primals_512.data_ptr()))
    del buf1080
    del buf1088
    del buf1096
    del buf1104
    del buf1112
    del buf1120
    del buf1128
    del buf1136
    del buf1144
    del buf1152
    del buf1160
    del buf1168
    del buf1176
    del buf1184
    del buf1192
    del buf1200
    del buf1208
    del buf1216
    del buf1224
    del primals_457
    del primals_458
    del primals_459
    del primals_460
    del primals_461
    del primals_462
    del primals_463
    del primals_464
    del primals_465
    del primals_466
    del primals_467
    del primals_468
    del primals_469
    del primals_470
    del primals_471
    del primals_472
    del primals_473
    del primals_474
    del primals_475
    del primals_476
    del primals_477
    del primals_478
    del primals_479
    del primals_480
    del primals_481
    del primals_482
    del primals_483
    del primals_484
    del primals_485
    del primals_486
    del primals_487
    del primals_488
    del primals_489
    del primals_490
    del primals_491
    del primals_492
    del primals_493
    del primals_494
    del primals_495
    del primals_496
    del primals_497
    del primals_498
    del primals_499
    del primals_500
    del primals_501
    del primals_502
    del primals_503
    del primals_504
    del primals_505
    del primals_506
    del primals_507
    del primals_508
    del primals_509
    del primals_510
    del primals_511
    del primals_512
    return (buf562, primals_1, buf0, primals_6, primals_8, primals_9, primals_11, primals_12, primals_14, primals_15, primals_17, primals_18, primals_20, primals_21, primals_23, primals_24, primals_26, primals_27, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_48, primals_50, primals_51, primals_53, primals_54, primals_56, primals_57, primals_59, primals_60, primals_62, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_170, primals_171, primals_173, primals_174, primals_176, primals_177, primals_179, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_234, primals_236, primals_237, primals_239, primals_240, primals_242, primals_243, primals_245, primals_246, primals_248, primals_249, primals_251, primals_252, primals_254, primals_255, primals_257, primals_258, primals_260, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, buf1, buf2, buf6, buf7, buf8, buf12, buf13, buf14, buf18, buf21, buf22, buf26, buf27, buf28, buf32, buf33, buf34, buf38, buf39, buf40, buf44, buf47, buf48, buf52, buf53, buf54, buf58, buf59, buf60, buf64, buf65, buf69, buf70, buf71, buf75, buf76, buf77, buf81, buf82, buf83, buf87, buf90, buf91, buf95, buf96, buf97, buf101, buf102, buf103, buf107, buf108, buf109, buf113, buf116, buf117, buf121, buf122, buf124, buf126, buf128, buf129, buf130, buf134, buf135, buf136, buf140, buf141, buf145, buf146, buf147, buf151, buf152, buf153, buf157, buf158, buf159, buf163, buf166, buf168, buf170, buf172, buf173, buf174, buf178, buf179, buf180, buf184, buf185, buf186, buf190, buf191, buf192, buf196, buf199, buf200, buf204, buf205, buf206, buf210, buf211, buf212, buf216, buf217, buf221, buf222, buf223, buf227, buf228, buf229, buf233, buf234, buf235, buf239, buf242, buf243, buf247, buf248, buf249, buf253, buf254, buf255, buf259, buf260, buf261, buf265, buf268, buf269, buf273, buf274, buf275, buf279, buf280, buf281, buf285, buf286, buf287, buf291, buf294, buf295, buf299, buf300, buf301, buf305, buf306, buf307, buf311, buf312, buf313, buf317, buf320, buf322, buf324, buf326, buf327, buf328, buf332, buf333, buf334, buf338, buf339, buf343, buf344, buf345, buf349, buf350, buf351, buf355, buf356, buf357, buf361, buf364, buf366, buf368, buf370, buf371, buf372, buf376, buf377, buf378, buf382, buf383, buf384, buf388, buf389, buf390, buf394, buf397, buf398, buf402, buf403, buf405, buf407, buf409, buf410, buf411, buf415, buf416, buf417, buf421, buf422, buf426, buf427, buf428, buf432, buf433, buf434, buf438, buf439, buf440, buf444, buf447, buf448, buf452, buf453, buf454, buf458, buf459, buf460, buf464, buf465, buf466, buf470, buf473, buf475, buf477, buf479, buf480, buf481, buf485, buf486, buf487, buf491, buf492, buf493, buf497, buf498, buf499, buf503, buf506, buf507, buf511, buf512, buf513, buf517, buf518, buf519, buf523, buf524, buf525, buf529, buf532, buf534, buf536, buf538, buf539, buf540, buf544, buf545, buf546, buf550, buf551, buf552, buf556, buf559, buf561, reinterpret_tensor(primals_3, (1000, 1280), (1280, 1), 0), buf563, buf564, reinterpret_tensor(buf553, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf547, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf541, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf565, buf566, reinterpret_tensor(buf526, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf520, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf514, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf508, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf567, reinterpret_tensor(buf500, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf494, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf488, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf482, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf568, buf569, reinterpret_tensor(buf467, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf461, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf455, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf449, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf570, reinterpret_tensor(buf441, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf435, (1, 480, 1, 1), (480, 1, 1, 1), 0), reinterpret_tensor(buf429, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf423, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf418, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf412, (1, 80, 1, 1), (80, 1, 1, 1), 0), buf571, reinterpret_tensor(buf399, (1, 672, 1, 1), (672, 1, 1, 1), 0), buf572, reinterpret_tensor(buf391, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf385, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf379, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf373, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf573, buf574, reinterpret_tensor(buf358, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf352, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf346, (1, 112, 1, 1), (112, 1, 1, 1), 0), reinterpret_tensor(buf340, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf335, (1, 56, 1, 1), (56, 1, 1, 1), 0), reinterpret_tensor(buf329, (1, 56, 1, 1), (56, 1, 1, 1), 0), buf575, buf576, reinterpret_tensor(buf314, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf308, (1, 240, 1, 1), (240, 1, 1, 1), 0), reinterpret_tensor(buf302, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf296, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf577, reinterpret_tensor(buf288, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf282, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf276, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf270, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf578, reinterpret_tensor(buf262, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf256, (1, 92, 1, 1), (92, 1, 1, 1), 0), reinterpret_tensor(buf250, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf244, (1, 40, 1, 1), (40, 1, 1, 1), 0), buf579, reinterpret_tensor(buf236, (1, 100, 1, 1), (100, 1, 1, 1), 0), reinterpret_tensor(buf230, (1, 100, 1, 1), (100, 1, 1, 1), 0), reinterpret_tensor(buf224, (1, 80, 1, 1), (80, 1, 1, 1), 0), reinterpret_tensor(buf218, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf213, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf207, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf201, (1, 240, 1, 1), (240, 1, 1, 1), 0), buf580, reinterpret_tensor(buf193, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf187, (1, 120, 1, 1), (120, 1, 1, 1), 0), reinterpret_tensor(buf181, (1, 20, 1, 1), (20, 1, 1, 1), 0), reinterpret_tensor(buf175, (1, 20, 1, 1), (20, 1, 1, 1), 0), buf581, buf582, reinterpret_tensor(buf160, (1, 60, 1, 1), (60, 1, 1, 1), 0), reinterpret_tensor(buf154, (1, 60, 1, 1), (60, 1, 1, 1), 0), reinterpret_tensor(buf148, (1, 40, 1, 1), (40, 1, 1, 1), 0), reinterpret_tensor(buf142, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf137, (1, 20, 1, 1), (20, 1, 1, 1), 0), reinterpret_tensor(buf131, (1, 20, 1, 1), (20, 1, 1, 1), 0), buf583, reinterpret_tensor(buf118, (1, 72, 1, 1), (72, 1, 1, 1), 0), buf584, reinterpret_tensor(buf110, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf104, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf98, (1, 12, 1, 1), (12, 1, 1, 1), 0), reinterpret_tensor(buf92, (1, 12, 1, 1), (12, 1, 1, 1), 0), buf585, reinterpret_tensor(buf84, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf78, (1, 36, 1, 1), (36, 1, 1, 1), 0), reinterpret_tensor(buf72, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf66, (1, 16, 1, 1), (16, 1, 1, 1), 0), reinterpret_tensor(buf61, (1, 12, 1, 1), (12, 1, 1, 1), 0), reinterpret_tensor(buf55, (1, 12, 1, 1), (12, 1, 1, 1), 0), reinterpret_tensor(buf49, (1, 48, 1, 1), (48, 1, 1, 1), 0), buf586, reinterpret_tensor(buf41, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf35, (1, 24, 1, 1), (24, 1, 1, 1), 0), reinterpret_tensor(buf29, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf23, (1, 8, 1, 1), (8, 1, 1, 1), 0), buf587, reinterpret_tensor(buf15, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf9, (1, 8, 1, 1), (8, 1, 1, 1), 0), reinterpret_tensor(buf3, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((8, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((12, 48, 1, 1), (48, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((24, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((12, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((12, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((36, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((36, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((72, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((20, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((24, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((40, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((60, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((20, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((20, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((40, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((80, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((100, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((100, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((40, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((92, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((92, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((40, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((40, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((56, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((112, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((56, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((56, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((336, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((168, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((80, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((160, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((480, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((80, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_274 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_279 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_282 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_285 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_288 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_291 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_294 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_297 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_300 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_303 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_306 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_309 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_312 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_315 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_318 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_321 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((12, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_324 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_327 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((36, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_330 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_333 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_336 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_339 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_342 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_345 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_348 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((60, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_351 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_354 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_357 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_360 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_363 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_366 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_369 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_371 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_372 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_373 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_374 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_375 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_378 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_381 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((100, ), (1, ), device='cpu', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_384 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_387 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_390 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_391 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_393 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_394 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_396 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_397 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_399 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_400 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_402 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_403 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_405 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_406 = rand_strided((92, ), (1, ), device='cpu', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_408 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_409 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_411 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_412 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_414 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_415 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_416 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_417 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_418 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_419 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_420 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_421 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_422 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_423 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_424 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_425 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_426 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_427 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_428 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_429 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_430 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_431 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_432 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_433 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_434 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_435 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_436 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_437 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_438 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_439 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_440 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_441 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_442 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    primals_443 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_444 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_445 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_446 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_447 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_448 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    primals_449 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_450 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_451 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_453 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_454 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_456 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_457 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_459 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_460 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_462 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_463 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_465 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_466 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_468 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_469 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_471 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_472 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_474 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_475 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_477 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_478 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_480 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_481 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_483 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_484 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_486 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_487 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_489 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_490 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_492 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_493 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_495 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_496 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_498 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_499 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_501 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_502 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_504 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_505 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_507 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_508 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_510 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_511 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_513 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('ghostnet_100', benchmark_compiled_module)
